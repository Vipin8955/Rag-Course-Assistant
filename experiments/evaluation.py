"""
experiments/evaluation.py — Compare RAG configurations on speed and retrieval quality.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TEST_QUERIES: list[str] = [
    "What is the main topic of this course?",
    "Explain the key concepts covered.",
    "What are the learning objectives?",
    "Describe the assignment requirements.",
    "What topics are covered in the syllabus?",
]


def evaluate_configurations(
    docs: dict[str, str],
    test_queries: list[str],
    configurations: list[dict],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Evaluate different RAG configurations on a set of test queries.

    Args:
        docs:           {filename: text} mapping (must be non-empty).
        test_queries:   List of test questions (must be non-empty).
        configurations: List of config dicts with keys:
                        chunk_size, overlap, model_name, store_type.
        top_k:          Number of results to retrieve per query.

    Returns:
        DataFrame with one row per configuration and evaluation metrics.

    Raises:
        ValueError: If docs or test_queries are empty.
    """
    if not docs:
        raise ValueError("evaluate_configurations: docs must not be empty.")
    if not test_queries:
        raise ValueError("evaluate_configurations: test_queries must not be empty.")
    if not configurations:
        logger.warning("evaluate_configurations: no configurations provided — returning empty DataFrame.")
        return pd.DataFrame()

    # Lazy import to keep module light and avoid circular deps
    from src.retriever import build_index, retrieve

    results: list[dict] = []

    for cfg_idx, cfg in enumerate(configurations):
        # ── Extract and validate config ───────────────────────────────────────
        chunk_size = cfg.get("chunk_size")
        overlap = cfg.get("overlap", 75)
        model_name = cfg.get("model_name")
        store_type = cfg.get("store_type")

        if not all([chunk_size, model_name, store_type]):
            logger.warning(
                "Config #%d is missing required keys (chunk_size/model_name/store_type) — skipping.",
                cfg_idx,
            )
            continue

        config_label = f"{model_name} | {store_type} | {chunk_size}w"
        logger.info("Evaluating config: %s", config_label)

        # ── Build index ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            store, chunks, stats = build_index(
                docs=docs,
                chunk_size=chunk_size,
                overlap=overlap,
                model_name=model_name,
                store_type=store_type,
            )
            index_time = time.perf_counter() - t0
        except Exception as e:
            logger.error("Index build failed for config '%s': %s", config_label, e)
            results.append(
                {
                    "Configuration": config_label,
                    "Chunk Size": chunk_size,
                    "Embedding Model": model_name,
                    "Vector DB": store_type,
                    "Total Chunks": 0,
                    "Avg Chunk Words": 0,
                    "Index Time (s)": 0.0,
                    "Avg Retrieval Time (ms)": 0.0,
                    "Avg Top Score (%)": 0.0,
                    "Avg Score Gap (%)": 0.0,
                    "Error": str(e),
                }
            )
            continue

        if store is None:
            logger.warning("Config '%s' produced no indexed chunks — skipping queries.", config_label)
            results.append(
                {
                    "Configuration": config_label,
                    "Chunk Size": chunk_size,
                    "Embedding Model": model_name,
                    "Vector DB": store_type,
                    "Total Chunks": 0,
                    "Avg Chunk Words": 0,
                    "Index Time (s)": round(index_time, 2),
                    "Avg Retrieval Time (ms)": 0.0,
                    "Avg Top Score (%)": 0.0,
                    "Avg Score Gap (%)": 0.0,
                    "Error": "No chunks produced",
                }
            )
            continue

        # ── Evaluate on queries ───────────────────────────────────────────────
        retrieval_times: list[float] = []
        top_scores: list[float] = []
        score_gaps: list[float] = []

        for query in test_queries:
            if not query or not query.strip():
                logger.debug("Skipping empty test query.")
                continue
            try:
                t1 = time.perf_counter()
                retrieved = retrieve(query, store, model_name=model_name, top_k=top_k)
                retrieval_times.append((time.perf_counter() - t1) * 1000)

                if retrieved:
                    scores = [c.get("score", 0.0) for c in retrieved]
                    top_scores.append(scores[0])
                    score_gaps.append(scores[0] - scores[-1] if len(scores) > 1 else 0.0)
                else:
                    top_scores.append(0.0)
                    score_gaps.append(0.0)

            except Exception as e:
                logger.warning("Query failed for config '%s': %s — skipping query.", config_label, e)
                top_scores.append(0.0)
                score_gaps.append(0.0)

        def _safe_mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        results.append(
            {
                "Configuration": config_label,
                "Chunk Size": chunk_size,
                "Embedding Model": model_name,
                "Vector DB": store_type,
                "Total Chunks": stats.get("total", 0),
                "Avg Chunk Words": stats.get("avg_words", 0),
                "Index Time (s)": round(index_time, 2),
                "Avg Retrieval Time (ms)": round(_safe_mean(retrieval_times), 1),
                "Avg Top Score (%)": round(_safe_mean(top_scores) * 100, 1),
                "Avg Score Gap (%)": round(_safe_mean(score_gaps) * 100, 1),
            }
        )
        logger.info(
            "Config '%s': %d chunks, index=%.1fs, avg_score=%.1f%%.",
            config_label,
            stats.get("total", 0),
            index_time,
            _safe_mean(top_scores) * 100,
        )

    if not results:
        logger.warning("evaluate_configurations: no results produced.")
        return pd.DataFrame()

    return pd.DataFrame(results)
