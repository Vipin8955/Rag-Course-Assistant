"""
app.py — University Course Assistant
A RAG-powered Q&A system for students to query their uploaded course materials.
"""

import logging
import os
import sys
import time

import streamlit as st
import pandas as pd

# Ensure src/ and experiments/ are importable BEFORE any local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Load .env + set env vars (must happen before torch/faiss import) ──────────
from src.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_STORE,
)

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="University Course Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Dark Academic Theme) ─────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9d1d9;
}

/* ── Header ── */
.hero-header {
    background: linear-gradient(135deg, #1a2332 0%, #0d1117 50%, #1a1a2e 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(88, 166, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(163, 113, 247, 0.06) 0%, transparent 50%);
    pointer-events: none;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #58a6ff, #a371f7, #f778ba);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
}

.hero-sub {
    color: #8b949e;
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.02em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #30363d;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.card-accent {
    background: linear-gradient(135deg, #1c2333, #161b22);
    border: 1px solid #388bfd40;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #58a6ff;
    margin: 0;
}

.metric-label {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0;
}

/* ── Chat ── */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 1rem;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 12px;
    margin-bottom: 1rem;
    scroll-behavior: smooth;
}

.chat-bubble-user {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #ffffff;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 0.9rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(56, 139, 253, 0.25);
}

.chat-bubble-assistant {
    background: #1c2333;
    border: 1px solid #30363d;
    color: #c9d1d9;
    border-radius: 16px 16px 16px 4px;
    padding: 0.75rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.chat-meta {
    font-size: 0.72rem;
    color: #8b949e;
    margin-top: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

.chat-role-user { text-align: right; }
.chat-role-assistant { text-align: left; }

/* ── Chunk display ── */
.chunk-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #388bfd;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}

.chunk-source {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #58a6ff;
    margin-bottom: 0.4rem;
}

.chunk-score {
    display: inline-block;
    background: #1f6feb20;
    color: #58a6ff;
    border-radius: 4px;
    padding: 0.1rem 0.4rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin-left: 0.5rem;
}

.chunk-text {
    color: #8b949e;
    line-height: 1.6;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

.badge-green { background: #1a3a2a; color: #56d364; border: 1px solid #2ea04320; }
.badge-blue  { background: #1a2a3a; color: #58a6ff; border: 1px solid #1f6feb20; }
.badge-purple{ background: #2a1a3a; color: #d2a8ff; border: 1px solid #a371f720; }
.badge-red   { background: #3a1a1a; color: #f85149; border: 1px solid #f8514920; }

/* ── Inputs & Buttons ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.15) !important;
}

.stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}

.stSlider > div {
    color: #58a6ff !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(56, 139, 253, 0.35) !important;
}

.stButton > button[kind="secondary"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
}

/* ── File uploader ── */
.stFileUploader {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
    transition: border-color 0.2s ease;
}

.stFileUploader:hover {
    border-color: #388bfd !important;
}

/* ── Dividers ── */
hr { border-color: #30363d !important; }

/* ── DataFrames ── */
.stDataFrame {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #c9d1d9 !important;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1f6feb, #a371f7) !important;
    border-radius: 4px !important;
}

/* ── Alerts ── */
.stSuccess { background: #1a3a2a !important; border: 1px solid #2ea04340 !important; }
.stWarning { background: #3a2a1a !important; border: 1px solid #d29922 !important; }
.stError   { background: #3a1a1a !important; border: 1px solid #f85149 !important; }
.stInfo    { background: #1a2a3a !important; border: 1px solid #1f6feb40 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #8b949e; }

/* ── Section headers ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #e6edf3;
    margin: 0 0 0.2rem 0;
}

.section-subtitle {
    color: #8b949e;
    font-size: 0.83rem;
    margin: 0 0 1rem 0;
}

/* ── File list ── */
.file-item {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.88rem;
    color: #c9d1d9;
}

.dot-green { color: #56d364; }
.dot-blue  { color: #58a6ff; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Imports ───────────────────────────────────────────────────────────────────
from src.ingestion import save_uploaded_file, list_uploaded_files, load_all_documents, delete_file
from src.chunking import chunk_documents, get_chunk_stats
from src.embeddings import AVAILABLE_MODELS, embed_texts, get_embedding_dim
from src.vector_store import create_store
from src.retriever import build_index, retrieve
from src.prompt import build_prompt, NOT_FOUND_MSG
from src.generator import generate_answer, AVAILABLE_LLMS, DEFAULT_LLM
from src.utils import format_score, truncate_text, get_index_key, timestamp
from experiments.evaluation import evaluate_configurations, DEFAULT_TEST_QUERIES

# ── Session State Init ────────────────────────────────────────────────────────
def init_session() -> None:
    """Initialise all session state keys with safe defaults."""
    defaults = {
        "chat_history": [],
        "vector_store": None,
        "index_key": "",
        "index_stats": {},
        "docs_loaded": {},
        "chunk_list": [],
        "eval_results": None,
        "indexing_done": False,
        "last_retrieved": [],
        "index_error": None,     # stores last indexing error message
        "query_error": None,     # stores last query error message
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <p class="hero-title">🎓 University Course Assistant</p>
    <p class="hero-sub">
        Upload your course PDFs · Ask anything · Get answers grounded in your documents
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**✂️ Chunk Settings**")
    _chunk_idx = [300, 700].index(DEFAULT_CHUNK_SIZE) if DEFAULT_CHUNK_SIZE in [300, 700] else 0
    chunk_size = st.selectbox(
        "Chunk Size (words)",
        options=[300, 700],
        index=_chunk_idx,
        help="Smaller chunks = more precise; Larger = more context",
    )
    overlap = st.slider(
        "Overlap (words)",
        min_value=50,
        max_value=100,
        value=min(max(DEFAULT_OVERLAP, 50), 100),
        step=25,
    )

    st.markdown("---")
    st.markdown("**🧠 Embedding Model**")
    _model_keys = list(AVAILABLE_MODELS.keys())
    _model_idx = _model_keys.index(DEFAULT_EMBEDDING_MODEL) if DEFAULT_EMBEDDING_MODEL in _model_keys else 0
    model_name = st.selectbox(
        "Embedding Model",
        options=_model_keys,
        index=_model_idx,
    )

    st.markdown("---")
    st.markdown("**🗄️ Vector Database**")
    _store_idx = ["FAISS", "ChromaDB"].index(DEFAULT_VECTOR_STORE) if DEFAULT_VECTOR_STORE in ["FAISS", "ChromaDB"] else 0
    store_type = st.selectbox(
        "Vector Store",
        options=["FAISS", "ChromaDB"],
        index=_store_idx,
    )

    st.markdown("---")
    st.markdown("**🤖 Language Model**")
    _llm_keys = list(AVAILABLE_LLMS.keys())
    _llm_idx = _llm_keys.index(DEFAULT_LLM) if DEFAULT_LLM in _llm_keys else 0
    llm_name = st.selectbox(
        "LLM",
        options=_llm_keys,
        index=_llm_idx,
        help="FLAN-T5-Base is fastest; FLAN-T5-Large is more accurate",
    )

    st.markdown("---")
    st.markdown("**🔍 Retrieval Settings**")
    top_k = st.slider("Top-K Results", min_value=2, max_value=10, value=min(max(DEFAULT_TOP_K, 2), 10))
    show_chunks = st.toggle("Show Retrieved Chunks", value=False)

    st.markdown("---")
    # Index status
    if st.session_state.indexing_done:
        stats = st.session_state.index_stats
        st.markdown(
            f"""
<div class="card">
<p style="margin:0;font-size:0.8rem;color:#56d364;font-weight:600;">✅ Index Ready</p>
<p style="margin:0.3rem 0 0;font-size:0.78rem;color:#8b949e;">{stats.get('total', 0)} chunks indexed</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="card">
<p style="margin:0;font-size:0.8rem;color:#f85149;font-weight:600;">⚠️ Not Indexed</p>
<p style="margin:0.3rem 0 0;font-size:0.78rem;color:#8b949e;">Upload docs and build index</p>
</div>
""",
            unsafe_allow_html=True,
        )

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📄  Upload Documents", "💬  Ask Questions", "📊  Evaluation", "❓  Help"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Upload Documents
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.3, 1], gap="large")

    with col_left:
        st.markdown('<p class="section-title">Upload Course PDFs</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Syllabus, lecture notes, assignments — any PDF</p>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            saved, failed = [], []
            for uf in uploaded_files:
                try:
                    path = save_uploaded_file(uf)
                    if path:
                        saved.append(uf.name)
                    else:
                        failed.append(uf.name)
                except Exception as e:
                    logger.error("Error saving '%s': %s", uf.name, e)
                    failed.append(uf.name)
            if saved:
                st.success(f"✅ Saved {len(saved)} file(s): {', '.join(saved)}")
            if failed:
                st.error(f"❌ Failed to save: {', '.join(failed)} — check file integrity or disk space.")

        st.markdown("---")

        # Build Index Button
        existing_files = list_uploaded_files()
        if existing_files:
            if st.button("⚡ Build Index", use_container_width=True):
                st.session_state.index_error = None
                progress = st.progress(0, text="Loading documents…")
                try:
                    with st.spinner("📚 Loading documents..."):
                        docs = load_all_documents()
                        if not docs:
                            st.warning("⚠️ No text could be extracted from the uploaded PDFs. "
                                       "Ensure they are text-based (not scanned images).")
                            progress.empty()
                            st.stop()
                        st.session_state.docs_loaded = docs

                    progress.progress(10, text="Chunking documents…")
                    time.sleep(0.2)

                    with st.spinner("✂️ Chunking text…"):
                        chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
                        st.session_state.chunk_list = chunks
                        stats = get_chunk_stats(chunks)

                    if not chunks:
                        st.warning("⚠️ No chunks produced — try a smaller chunk size or check your PDFs.")
                        progress.empty()
                        st.stop()

                    progress.progress(33, text="Generating embeddings…")

                    with st.spinner(f"🧠 Embedding with {model_name}…"):
                        texts = [c["text"] for c in chunks]
                        dim = get_embedding_dim(model_name)
                        embeddings = embed_texts(texts, model_name=model_name, show_progress=False)

                    progress.progress(66, text="Building vector index…")

                    with st.spinner(f"🗄️ Indexing into {store_type}…"):
                        store = create_store(store_type, dim=dim)
                        store.add(embeddings, chunks)
                        st.session_state.vector_store = store
                        st.session_state.index_stats = stats
                        st.session_state.indexing_done = True

                    progress.progress(100, text="Done!")
                    time.sleep(0.5)
                    progress.empty()
                    st.success(
                        f"🎉 Index built! {stats['total']} chunks from {len(docs)} document(s)"
                    )

                except (ImportError, RuntimeError) as e:
                    progress.empty()
                    err = str(e)
                    st.session_state.index_error = err
                    logger.error("Index build failed: %s", err)
                    st.error(f"❌ Indexing failed: {err}")
                except MemoryError:
                    progress.empty()
                    msg = "Out of memory. Try using FAISS + MiniLM with 300-word chunks, or upload fewer documents."
                    st.session_state.index_error = msg
                    st.error(f"❌ {msg}")
                except Exception as e:
                    progress.empty()
                    logger.exception("Unexpected indexing error.")
                    st.error(f"❌ Unexpected error: {e}")
        else:
            st.info("👆 Upload at least one PDF to get started.")

    with col_right:
        st.markdown('<p class="section-title">Indexed Documents</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-subtitle">Files in your knowledge base</p>',
            unsafe_allow_html=True,
        )

        existing_files = list_uploaded_files()
        if existing_files:
            for fname in existing_files:
                col_f, col_del = st.columns([4, 1])
                with col_f:
                    st.markdown(
                        f"""
<div class="file-item">
    <span class="dot-green">●</span>
    <span>{fname}</span>
</div>
""",
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_{fname}", help=f"Delete {fname}"):
                        try:
                            deleted = delete_file(fname)
                            if deleted:
                                st.session_state.indexing_done = False
                                st.session_state.vector_store = None
                                st.session_state.docs_loaded = {}
                            else:
                                st.warning(f"Could not delete '{fname}' — file may already be removed.")
                        except Exception as e:
                            logger.error("Error deleting '%s': %s", fname, e)
                            st.error(f"Failed to delete '{fname}': {e}")
                        st.rerun()
        else:
            st.markdown(
                """
<div class="card" style="text-align:center;padding:2rem;">
    <p style="font-size:2.5rem;margin:0">📂</p>
    <p style="color:#8b949e;margin:0.5rem 0 0;">No documents yet</p>
</div>
""",
                unsafe_allow_html=True,
            )

        if st.session_state.indexing_done:
            st.markdown("---")
            st.markdown("**📊 Index Statistics**")
            stats = st.session_state.index_stats
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><p class="metric-value">{stats.get("total", 0)}</p><p class="metric-label">Chunks</p></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><p class="metric-value">{stats.get("avg_words", 0)}</p><p class="metric-label">Avg Words</p></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><p class="metric-value">{len(existing_files)}</p><p class="metric-label">Documents</p></div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Ask Questions (Chat)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.indexing_done:
        st.markdown(
            """
<div class="card-accent" style="text-align:center;padding:3rem;">
    <p style="font-size:3rem;margin:0">🔒</p>
    <p style="font-size:1.1rem;font-weight:600;color:#e6edf3;margin:0.8rem 0 0.3rem;">Index Not Built</p>
    <p style="color:#8b949e;margin:0;">Please upload documents and build the index first (Tab 1).</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        chat_col, info_col = st.columns([1.6, 1], gap="large")

        with chat_col:
            st.markdown('<p class="section-title">💬 Chat with Your Documents</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-subtitle">Ask any question about your uploaded course materials</p>',
                unsafe_allow_html=True,
            )

            # Chat history display
            chat_html = '<div class="chat-container" id="chat-box">'

            if not st.session_state.chat_history:
                chat_html += """
<div style="text-align:center;padding:3rem 1rem;color:#8b949e;">
    <p style="font-size:2rem;margin:0">🤖</p>
    <p style="margin:0.5rem 0 0;font-size:0.9rem;">
        Hi! Ask me anything about your course documents.
    </p>
    <p style="margin:0.3rem 0 0;font-size:0.8rem;color:#6e7681;">
        Try: "Explain Unit 1" · "What is covered in this course?" · "Summarise Chapter 2"
    </p>
</div>"""
            else:
                for msg in st.session_state.chat_history:
                    role = msg["role"]
                    content = msg["content"].replace("\n", "<br>")
                    ts = msg.get("time", "")
                    if role == "user":
                        chat_html += f"""
<div class="chat-role-user">
    <div class="chat-bubble-user">{content}</div>
    <div class="chat-meta">{ts}</div>
</div>"""
                    else:
                        chat_html += f"""
<div class="chat-role-assistant">
    <div class="chat-bubble-assistant">{content}</div>
    <div class="chat-meta">🤖 Assistant · {ts}</div>
</div>"""

            chat_html += "</div>"
            st.markdown(chat_html, unsafe_allow_html=True)

            # Auto-scroll JS
            st.markdown(
                """
<script>
const chatBox = document.getElementById('chat-box');
if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
</script>
""",
                unsafe_allow_html=True,
            )

            # Input row
            input_col, btn_col = st.columns([5, 1])
            with input_col:
                user_query = st.text_input(
                    "Ask a question",
                    placeholder="e.g. What is Machine Learning? / Summarise Chapter 2",
                    label_visibility="collapsed",
                    key="chat_input",
                )
            with btn_col:
                send = st.button("Send →", use_container_width=True)

            # Clear chat
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat", type="secondary"):
                    st.session_state.chat_history = []
                    st.rerun()

            # Handle send
            if send and user_query.strip():
                query = user_query.strip()
                st.session_state.chat_history.append(
                    {"role": "user", "content": query, "time": timestamp()}
                )
                try:
                    with st.spinner("🔍 Searching documents…"):
                        retrieved = retrieve(
                            query,
                            st.session_state.vector_store,
                            model_name=model_name,
                            top_k=top_k,
                        )
                    with st.spinner("🤖 Generating answer…"):
                        prompt = build_prompt(query, retrieved)
                        answer = generate_answer(
                            prompt, model_name=llm_name, retrieved_chunks=retrieved
                        )
                    st.session_state.last_retrieved = retrieved
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "time": timestamp()}
                    )
                except (RuntimeError, ImportError) as e:
                    logger.error("Query pipeline error: %s", e)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"⚠️ Error: {e}", "time": timestamp()}
                    )
                except Exception as e:
                    logger.exception("Unexpected query error.")
                    st.session_state.chat_history.append(
                        {"role": "assistant",
                         "content": f"⚠️ Unexpected error: {e}",
                         "time": timestamp()}
                    )
                st.rerun()

            # Example prompts
            st.markdown("**💡 Quick Prompts**")
            qp_cols = st.columns(3)
            example_prompts = [
                "Explain Unit 1",
                "What is Machine Learning?",
                "Summarise Chapter 2",
                "List the assignments",
                "What are the prerequisites?",
                "Explain key concepts",
            ]
            for i, prompt_ex in enumerate(example_prompts):
                with qp_cols[i % 3]:
                    if st.button(prompt_ex, key=f"qp_{i}", use_container_width=True, type="secondary"):
                        st.session_state.chat_history.append(
                            {"role": "user", "content": prompt_ex, "time": timestamp()}
                        )
                        try:
                            with st.spinner("Processing…"):
                                retrieved = retrieve(
                                    prompt_ex,
                                    st.session_state.vector_store,
                                    model_name=model_name,
                                    top_k=top_k,
                                )
                                prompt_built = build_prompt(prompt_ex, retrieved)
                                answer = generate_answer(
                                    prompt_built, model_name=llm_name, retrieved_chunks=retrieved
                                )
                            st.session_state.last_retrieved = retrieved
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": answer, "time": timestamp()}
                            )
                        except Exception as e:
                            logger.error("Quick prompt error: %s", e)
                            st.session_state.chat_history.append(
                                {"role": "assistant",
                                 "content": f"⚠️ Error processing prompt: {e}",
                                 "time": timestamp()}
                            )
                        st.rerun()

        with info_col:
            st.markdown('<p class="section-title">🔍 Context Used</p>', unsafe_allow_html=True)
            st.markdown(
                '<p class="section-subtitle">Chunks retrieved for the last question</p>',
                unsafe_allow_html=True,
            )

            last_retrieved = st.session_state.get("last_retrieved", [])
            if last_retrieved and show_chunks:
                for i, chunk in enumerate(last_retrieved, 1):
                    source = chunk.get("source", "unknown")
                    score = chunk.get("score", 0)
                    text = truncate_text(chunk.get("text", ""), max_words=50)
                    wc = chunk.get("word_count", 0)

                    st.markdown(
                        f"""
<div class="chunk-card">
    <div class="chunk-source">
        📄 {source}
        <span class="chunk-score">Score: {format_score(score)}</span>
    </div>
    <div class="chunk-text">{text}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
            elif last_retrieved and not show_chunks:
                st.markdown(
                    f"""
<div class="card" style="text-align:center;">
    <p style="font-size:1.5rem;margin:0">🔍</p>
    <p style="color:#58a6ff;font-weight:600;margin:0.3rem 0 0;">{len(last_retrieved)} chunks found</p>
    <p style="color:#8b949e;font-size:0.8rem;margin:0.2rem 0 0;">Enable "Show Retrieved Chunks" in sidebar to view</p>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
<div class="card" style="text-align:center;padding:2rem;">
    <p style="font-size:2rem;margin:0">💭</p>
    <p style="color:#8b949e;margin:0.5rem 0 0;font-size:0.85rem;">Ask a question to see retrieved context here</p>
</div>
""",
                    unsafe_allow_html=True,
                )

            # Config summary
            st.markdown("---")
            st.markdown("**Current Config**")
            cfg_items = [
                ("🧠 Embedding", model_name),
                ("🗄️ Vector DB", store_type),
                ("✂️ Chunk Size", f"{chunk_size} words"),
                ("🤖 LLM", llm_name.split("(")[0].strip()),
                ("🔍 Top-K", str(top_k)),
            ]
            for label, val in cfg_items:
                st.markdown(
                    f"""
<div style="display:flex;justify-content:space-between;padding:0.3rem 0;border-bottom:1px solid #21262d;font-size:0.82rem;">
    <span style="color:#8b949e;">{label}</span>
    <span style="color:#c9d1d9;font-weight:500;">{val}</span>
</div>
""",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">📊 Configuration Evaluation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-subtitle">Compare FAISS vs ChromaDB · MiniLM vs BGE · 300 vs 700 word chunks</p>',
        unsafe_allow_html=True,
    )

    docs_for_eval = st.session_state.docs_loaded

    if not docs_for_eval:
        st.markdown(
            """
<div class="card-accent" style="text-align:center;padding:3rem;">
    <p style="font-size:3rem;margin:0">📂</p>
    <p style="color:#e6edf3;font-weight:600;margin:0.5rem 0 0.3rem;">No Documents Loaded</p>
    <p style="color:#8b949e;">Upload and index documents in Tab 1 first.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        # Test query input
        st.markdown("**🧪 Test Queries**")
        test_q_input = st.text_area(
            "Enter test queries (one per line)",
            value="\n".join(DEFAULT_TEST_QUERIES),
            height=130,
        )
        test_queries = [q.strip() for q in test_q_input.strip().split("\n") if q.strip()]

        # Configuration selection
        st.markdown("**⚙️ Configurations to Compare**")
        eval_col1, eval_col2 = st.columns(2)
        with eval_col1:
            eval_stores = st.multiselect(
                "Vector Stores", ["FAISS", "ChromaDB"], default=["FAISS", "ChromaDB"]
            )
            eval_models = st.multiselect(
                "Embedding Models",
                list(AVAILABLE_MODELS.keys()),
                default=["all-MiniLM-L6-v2"],
            )
        with eval_col2:
            eval_chunks = st.multiselect(
                "Chunk Sizes", [300, 700], default=[300, 700]
            )

        # Build config list
        configs = []
        for store in eval_stores:
            for model in eval_models:
                for cs in eval_chunks:
                    configs.append(
                        {
                            "store_type": store,
                            "model_name": model,
                            "chunk_size": cs,
                            "overlap": 75,
                        }
                    )

        st.markdown(f"**{len(configs)} configurations** will be evaluated on **{len(test_queries)} queries**")

        if st.button("🚀 Run Evaluation", use_container_width=True):
            if not configs:
                st.warning("Select at least one option from each setting.")
            elif not test_queries:
                st.warning("Add at least one test query before running evaluation.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_list = []

                for i, cfg in enumerate(configs):
                    label = f"{cfg['model_name']} | {cfg['store_type']} | {cfg['chunk_size']}w"
                    status_text.markdown(f"⏳ Evaluating: **{label}**")
                    try:
                        df = evaluate_configurations(
                            docs=docs_for_eval,
                            test_queries=test_queries,
                            configurations=[cfg],
                            top_k=top_k,
                        )
                        if not df.empty:
                            results_list.append(df)
                    except ValueError as e:
                        st.error(f"❌ Evaluation config error: {e}")
                        logger.error("Eval config error for '%s': %s", label, e)
                    except (RuntimeError, ImportError) as e:
                        st.warning(f"⚠️ Config failed: **{label}** — {e}")
                        logger.warning("Eval failed for '%s': %s", label, e)
                    except Exception as e:
                        logger.exception("Unexpected eval error for '%s'.", label)
                        st.warning(f"⚠️ Unexpected error for **{label}**: {e}")
                    progress_bar.progress((i + 1) / len(configs))

                status_text.markdown("✅ Evaluation complete!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()

                if results_list:
                    full_results = pd.concat(results_list, ignore_index=True)
                    st.session_state.eval_results = full_results
                else:
                    st.warning("No evaluation results produced — check that documents are indexed and configs are valid.")

        # Display results
        if st.session_state.eval_results is not None:
            df = st.session_state.eval_results
            st.markdown("---")
            st.markdown("**📈 Results**")

            # Style the DataFrame
            display_cols = [
                "Configuration", "Chunk Size", "Embedding Model", "Vector DB",
                "Total Chunks", "Avg Chunk Words", "Index Time (s)",
                "Avg Retrieval Time (ms)", "Avg Top Score (%)",
            ]
            available_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[available_cols].style.background_gradient(
                    subset=["Avg Top Score (%)"] if "Avg Top Score (%)" in df.columns else [],
                    cmap="Blues",
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Chart
            if "Avg Top Score (%)" in df.columns and "Configuration" in df.columns:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=df["Configuration"],
                        y=df["Avg Top Score (%)"],
                        marker=dict(
                            color=df["Avg Top Score (%)"],
                            colorscale="Blues",
                            showscale=False,
                        ),
                        text=df["Avg Top Score (%)"].apply(lambda x: f"{x:.1f}%"),
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    title="Average Top Similarity Score by Configuration",
                    paper_bgcolor="#0d1117",
                    plot_bgcolor="#161b22",
                    font=dict(family="DM Sans", color="#e6edf3"),
                    xaxis=dict(gridcolor="#30363d", tickangle=-20),
                    yaxis=dict(gridcolor="#30363d", title="Avg Top Score (%)"),
                    margin=dict(t=50, b=50, l=10, r=10),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Best config highlight
            if "Avg Top Score (%)" in df.columns:
                best = df.loc[df["Avg Top Score (%)"].idxmax()]
                st.markdown(
                    f"""
<div class="card-accent">
    <p style="color:#56d364;font-weight:700;margin:0 0 0.3rem;">🏆 Best Configuration</p>
    <p style="font-size:1.05rem;color:#e6edf3;margin:0 0 0.2rem;font-weight:600;">{best.get('Configuration', 'N/A')}</p>
    <p style="color:#8b949e;margin:0;font-size:0.85rem;">
        Avg Top Score: <b style="color:#58a6ff;">{best.get('Avg Top Score (%)', 0):.1f}%</b> · 
        Retrieval: <b style="color:#d2a8ff;">{best.get('Avg Retrieval Time (ms)', 0):.1f}ms</b>
    </p>
</div>
""",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Help
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">❓ Help & Documentation</p>', unsafe_allow_html=True)
    st.markdown("---")

    help_col1, help_col2 = st.columns(2)

    with help_col1:
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown(
                """
**Step 1 — Upload PDFs**
- Go to the **Upload Documents** tab
- Drag & drop your course PDFs (syllabus, notes, assignments)
- Files are saved to the `/data` directory

**Step 2 — Configure Settings**
- Use the sidebar to choose:
  - Chunk size (300 or 700 words)
  - Embedding model
  - Vector database (FAISS or ChromaDB)
  - Language model

**Step 3 — Build Index**
- Click **⚡ Build Index** to process your documents
- This chunks, embeds, and indexes all text

**Step 4 — Ask Questions**
- Switch to **Ask Questions** tab
- Type your question or click a quick prompt
- The system retrieves relevant chunks and generates an answer
"""
            )

        with st.expander("✂️ Chunk Size Guide"):
            st.markdown(
                """
| Size | Words | Best For |
|------|-------|----------|
| 300  | ~300  | Precise Q&A, definitions, specific facts |
| 700  | ~700  | Summaries, explanations, broad topics |

**Overlap** (50–100 words) ensures no information is lost at chunk boundaries.
"""
            )

        with st.expander("🧠 Embedding Models"):
            st.markdown(
                """
**all-MiniLM-L6-v2**
- Fast, lightweight (22M params)
- Good general-purpose semantic similarity
- Ideal for most course materials

**bge-base-en-v1.5** (BAAI)
- Larger, more accurate (109M params)
- Better at nuanced retrieval
- Slower but higher quality
"""
            )

    with help_col2:
        with st.expander("🗄️ Vector Databases"):
            st.markdown(
                """
**FAISS** (Facebook AI Similarity Search)
- In-memory, extremely fast
- Great for sessions with many queries
- Best for: quick experiments, smaller collections

**ChromaDB**
- Persistent storage
- Rich metadata filtering support
- Best for: larger document sets, production use
"""
            )

        with st.expander("🤖 Language Models"):
            st.markdown(
                """
**FLAN-T5-Base** *(recommended)*
- 250M parameters
- Fast inference on CPU
- Good instruction following

**FLAN-T5-Large**
- 780M parameters
- Better reasoning
- Requires more RAM (~3GB)

Both models answer ONLY from retrieved context — no hallucination from external knowledge.
"""
            )

        with st.expander("📊 Evaluation Guide"):
            st.markdown(
                """
The Evaluation tab compares different configurations automatically:

- **Avg Top Score**: Higher = better semantic match
- **Index Time**: How long it takes to build the index
- **Retrieval Time**: Speed of search per query
- **Total Chunks**: More chunks = finer granularity

**Tips:**
- Use FAISS + MiniLM for speed
- Use ChromaDB + BGE for accuracy
- 300-word chunks = better for specific questions
- 700-word chunks = better for summaries
"""
            )

        with st.expander("⚠️ Troubleshooting"):
            st.markdown(
                """
**"Answer not found in uploaded documents"**
→ The query doesn't match any content well. Try rephrasing.

**Slow indexing**
→ Large PDFs take longer. BGE model is slower than MiniLM.

**Model download errors**
→ First run requires internet to download models from HuggingFace.

**ChromaDB errors**
→ Try switching to FAISS, or delete the `data/chroma_db` folder.

**Out of memory**
→ Use FLAN-T5-Base (not Large) and FAISS (not ChromaDB).
"""
            )

    st.markdown("---")
    st.markdown(
        """
<div class="card" style="text-align:center;padding:1.5rem;">
    <p style="margin:0;color:#8b949e;font-size:0.85rem;">
        🎓 <b style="color:#c9d1d9;">University Course Assistant</b> · 
        Built with Streamlit + HuggingFace + FAISS + ChromaDB · 
        Answers strictly from your uploaded documents
    </p>
</div>
""",
        unsafe_allow_html=True,
    )
