"""
experiments/evaluation.py — Compare retrieval configurations
"""

from __future__ import annotations
import time
import random
import pandas as pd


def evaluate_configurations(
    docs: dict[str, str],
    test_queries: list[str],
    configurations: list[dict],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Evaluate different RAG configurations on test queries.

    Args:
        docs: {filename: text}
        test_queries: list of test questions
        configurations: list of config dicts with keys:
            chunk_size, overlap, model_name, store_type
        top_k: number of results to retrieve

    Returns:
        DataFrame with evaluation results
    """
    from src.retriever import build_index, retrieve
    from src.prompt import build_prompt
    from src.generator import generate_answer, DEFAULT_LLM

    results = []

    for cfg in configurations:
        chunk_size = cfg["chunk_size"]
        overlap = cfg.get("overlap", 75)
        model_name = cfg["model_name"]
        store_type = cfg["store_type"]

        config_label = f"{model_name} | {store_type} | {chunk_size}w"

        # Build index
        t0 = time.time()
        try:
            store, chunks, stats = build_index(
                docs=docs,
                chunk_size=chunk_size,
                overlap=overlap,
                model_name=model_name,
                store_type=store_type,
            )
            index_time = time.time() - t0
        except Exception as e:
            results.append(
                {
                    "Configuration": config_label,
                    "Total Chunks": 0,
                    "Avg Chunk Words": 0,
                    "Index Time (s)": 0,
                    "Avg Retrieval Time (ms)": 0,
                    "Avg Top Score (%)": 0,
                    "Avg Score Gap (%)": 0,
                    "Error": str(e),
                }
            )
            continue

        # Evaluate on queries
        retrieval_times = []
        top_scores = []
        score_gaps = []

        for query in test_queries:
            t1 = time.time()
            retrieved = retrieve(query, store, model_name=model_name, top_k=top_k)
            retrieval_times.append((time.time() - t1) * 1000)

            if retrieved:
                scores = [c.get("score", 0) for c in retrieved]
                top_scores.append(scores[0])
                if len(scores) > 1:
                    score_gaps.append(scores[0] - scores[-1])
                else:
                    score_gaps.append(0)
            else:
                top_scores.append(0)
                score_gaps.append(0)

        results.append(
            {
                "Configuration": config_label,
                "Chunk Size": chunk_size,
                "Embedding Model": model_name,
                "Vector DB": store_type,
                "Total Chunks": stats["total"],
                "Avg Chunk Words": stats["avg_words"],
                "Index Time (s)": round(index_time, 2),
                "Avg Retrieval Time (ms)": round(
                    sum(retrieval_times) / max(len(retrieval_times), 1), 1
                ),
                "Avg Top Score (%)": round(
                    sum(top_scores) / max(len(top_scores), 1) * 100, 1
                ),
                "Avg Score Gap (%)": round(
                    sum(score_gaps) / max(len(score_gaps), 1) * 100, 1
                ),
            }
        )

    return pd.DataFrame(results)


DEFAULT_TEST_QUERIES = [
    "What is the main topic of this course?",
    "Explain the key concepts covered.",
    "What are the learning objectives?",
    "Describe the assignment requirements.",
    "What topics are covered in the syllabus?",
]
