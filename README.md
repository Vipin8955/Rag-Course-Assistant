<div align="center">

# 🎓 University Course Assistant

**A production-ready Retrieval-Augmented Generation (RAG) system for students.**  
Upload your course PDFs · Ask anything · Get answers strictly grounded in your documents.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co)
[![FAISS](https://img.shields.io/badge/Facebook-FAISS-blue?logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?logoColor=white)](LICENSE)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=white)](https://render.com)

</div>

---

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Performance Benchmarks](#-performance-benchmarks)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration (.env)](#-configuration-env)
- [Usage Guide](#-usage-guide)
- [Configuration Options](#-configuration-options)
- [Troubleshooting](#-troubleshooting)
- [Design Decisions](#-key-design-decisions)

---

## 🤔 What It Does

The University Course Assistant lets students upload their **course PDFs** (lecture notes, syllabi, assignments) and ask natural language questions about them. The system retrieves the most relevant passages and generates answers **strictly grounded in what you uploaded** — no hallucinations, no external knowledge.

> **Think of it as a smart tutor that has read all your course material and can answer questions about it instantly — 100% offline, 100% private.**

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                           │
│                                                                     │
│  PDF Upload → Text Extraction → Word Chunking → Embedding Model     │
│   (PyMuPDF)     (fitz + clean)   (300/700 words)  (MiniLM/BGE)     │
│                                                       ↓             │
│                                              Vector Index           │
│                                         (FAISS or ChromaDB)        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
│                                                                     │
│  User Question → Embed Query → Semantic Search → Top-K Chunks       │
│                  (same model)   (cosine sim)    (configurable)      │
│                                                       ↓             │
│                                           Prompt Builder            │
│                                        (extractive QA format)      │
│                                                  ↓                  │
│                                        FLAN-T5 LLM → Answer         │
│                                    (Base 250M / Large 780M)         │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Ingest** — PDFs are parsed with PyMuPDF, cleaned of noise (page numbers, excessive whitespace, non-printable chars)
2. **Chunk** — Text split into overlapping word-based windows (300 or 700 words, configurable overlap)
3. **Embed** — Each chunk → 384-dim or 768-dim dense vector via sentence-transformers
4. **Index** — Vectors stored in FAISS (in-memory, fastest) or ChromaDB (disk-persistent)
5. **Retrieve** — At query time, the question is embedded and matched via L2-normalised cosine similarity
6. **Generate** — Top-K chunks are assembled into an extractive QA prompt and passed to FLAN-T5

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 **Multi-PDF Ingestion** | PyMuPDF extraction with cleaning — handles multi-page, multi-file |
| ✂️ **Smart Chunking** | Configurable 300 or 700 word chunks with 50–100 word overlap |
| 🧠 **Dual Embedding Models** | MiniLM-L6-v2 (fast, 22M params) or BGE-base-en-v1.5 (accurate, 109M params) |
| 🗄️ **Dual Vector Databases** | FAISS (in-memory, fastest) or ChromaDB (disk-persistent) |
| 🔍 **Semantic Retrieval** | L2-normalised cosine similarity, configurable Top-K (2–10) |
| 💬 **Chat Interface** | Multi-turn chat UI with history, timestamps, and dynamic quick prompts |
| 💡 **Dynamic Quick Prompts** | Auto-detects document type (resume/paper/course) and shows relevant questions |
| 🤖 **Local LLM** | FLAN-T5-Base or FLAN-T5-Large — 100% offline after first download |
| ⚠️ **Model Mismatch Guard** | Detects when sidebar model differs from indexed model, blocks stale queries |
| 📊 **Evaluation Tab** | Compare configs on indexing speed, retrieval speed, and similarity score |
| ⚙️ **Config via .env** | All defaults controllable from `.env` — no code changes needed |
| 🛡️ **Full Error Handling** | Graceful errors at every layer with user-friendly messages |
| ☁️ **Render Ready** | `render.yaml` + `.streamlit/config.toml` included for one-click deploy |

---

## 📊 Performance Benchmarks

> **Measured on:** Windows 11, Intel CPU, 16 GB RAM, Python 3.11  
> **Test corpus:** 1 PDF document (~2,200 chars extracted), 5 semantic queries

### Indexing & Retrieval Speed

| Configuration | Chunks | Index Time | Avg Retrieval | Similarity Score |
|---|:---:|---:|---:|---:|
| MiniLM-L6-v2 · FAISS · 300w | 11 | 509 ms\* | **13.5 ms** | 16.1% |
| MiniLM-L6-v2 · FAISS · 700w | 4 | **99 ms** | **12.5 ms** | 14.8% |
| MiniLM-L6-v2 · ChromaDB · 300w | 11 | 509 ms | 13.6 ms | 16.1% |
| BGE-base-en-v1.5 · FAISS · 300w | 11 | 760 ms | 29.9 ms | 53.3% |
| **BGE-base-en-v1.5 · FAISS · 700w** | 4 | 1,208 ms | 54.4 ms | **55.4%** |

> \* Indexing time excludes one-time model load (~4–7 seconds on first use, then cached in RAM).

### Key Takeaways

| Metric | Winner | Why |
|---|---|---|
| ⚡ Fastest indexing | MiniLM · FAISS · 700w | Fewer chunks, smaller model |
| ⚡ Fastest per-query | MiniLM · FAISS · 300w/700w | ~13 ms — near-instant retrieval |
| 🎯 Best accuracy | **BGE · FAISS · 700w** | BGE embeddings (768-dim) capture semantics much better |
| ⚖️ Best balance | MiniLM · FAISS · 300w | Good speed + finer-grained retrieval for Q&A |

**Recommended setup for most users:** `all-MiniLM-L6-v2 + FAISS + 300 words`  
**Recommended for accuracy:** `BGE-base-en-v1.5 + FAISS + 700 words`

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | [Streamlit 1.32](https://streamlit.io) | Interactive web interface |
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io) (`fitz`) | Robust text extraction |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) | Text → dense vectors |
| **Vector DB (fast)** | [FAISS](https://github.com/facebookresearch/faiss) | In-memory similarity search |
| **Vector DB (persistent)** | [ChromaDB](https://www.trychroma.com/) | Disk-based vector store |
| **LLM** | [FLAN-T5](https://huggingface.co/google/flan-t5-base) via HuggingFace | Answer generation |
| **Charts** | [Plotly](https://plotly.com/python/) | Evaluation visualisations |
| **Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) | `.env` file loading |

---

## 📁 Project Structure

```
rag_course_app/
│
├── app.py                    # Main Streamlit application (UI + routing)
├── render.yaml               # Render.com deployment config
├── requirements.txt          # Pinned Python dependencies
├── .env                      # Local config overrides (gitignored)
├── .env.example              # Template — copy to .env and customise
├── README.md
├── SETUP_GUIDE.md            # Step-by-step beginner setup guide
│
├── .streamlit/
│   └── config.toml           # Streamlit server config (port, headless)
│
├── src/                      # Core library modules
│   ├── config.py             # .env loader + centralised settings
│   ├── ingestion.py          # PDF save, extract, clean, load
│   ├── chunking.py           # Word-based text chunking with overlap
│   ├── embeddings.py         # Sentence embedding model wrappers
│   ├── vector_store.py       # FAISSStore + ChromaStore classes
│   ├── retriever.py          # Full index build + semantic retrieval
│   ├── prompt.py             # FLAN-T5 optimised extractive QA prompts
│   ├── generator.py          # FLAN-T5 text generation pipeline
│   └── utils.py              # Shared helpers (formatting, hashing, etc.)
│
├── experiments/
│   └── evaluation.py         # Config comparison & benchmarking framework
│
└── data/                     # Runtime data (gitignored except .gitkeep)
    └── .gitkeep              # Keeps data/ in git without contents
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9, 3.10, or 3.11** (recommended: 3.11)
- **4 GB RAM minimum** (8 GB recommended for FLAN-T5-Large)
- **~2 GB free disk space** (for model downloads on first run)
- **Internet connection** on first run only — models are cached locally after that

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rag_course_app.git
cd rag_course_app
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` in your prompt ✅

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This installs ~14 packages including PyTorch and may take 5–10 minutes.

**Windows — if `torch` is slow to install:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

Edit `.env` to change default models, chunk sizes, or directories.  
The app works with defaults — this step is optional.

### 5. Run the App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** 🎉

> 🌐 **First run:** Models (~90–440 MB) download automatically from HuggingFace and are cached locally. Subsequent starts are instant.

---

## 🔧 Configuration (.env)

Copy `.env.example` to `.env` and adjust:

```env
# Directories
DATA_DIR=data
CHROMA_DIR=data/chroma_db

# Default sidebar settings
DEFAULT_CHUNK_SIZE=300           # 300 or 700 words
DEFAULT_OVERLAP=75               # 50, 75, or 100 words
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2   # or bge-base-en-v1.5
DEFAULT_VECTOR_STORE=FAISS       # FAISS or ChromaDB
DEFAULT_LLM=FLAN-T5-Base (Fast)  # or FLAN-T5-Large
DEFAULT_TOP_K=5                  # 2–10 chunks retrieved per query

# Optional: HuggingFace token for private/gated models
# HF_TOKEN=hf_your_token_here
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`. Use `.env.example` as the shared template.

---

## 📖 Usage Guide

### Step 1 — Upload Documents
1. Open the **📄 Upload Documents** tab
2. Drag & drop your PDFs (syllabus, lecture notes, assignments)
3. Click **⚡ Build Index** — takes a few seconds

### Step 2 — Ask Questions
1. Switch to the **💬 Ask Questions** tab
2. Type your question or click a **Quick Prompt** button
   - Quick Prompts auto-update based on your document type (resume, research paper, or course notes)
3. The assistant retrieves relevant chunks and generates a grounded answer

### Step 3 — Evaluate Configurations (Optional)
1. Go to the **📊 Evaluation** tab
2. Select embedding models, vector stores, and chunk sizes to compare
3. Run evaluation to see speed vs. accuracy trade-offs

### ⚠️ Changing Models After Indexing

If you change the **Embedding Model** or **Vector Store** in the sidebar after building the index, the app will:
- Show an **amber warning banner** explaining exactly what changed
- **Disable** the Send button and Quick Prompts until you rebuild

To fix: go back to Tab 1 and click **⚡ Build Index** again.

---

## ⚙️ Configuration Options

### Chunk Size

| Size | Words | Best For |
|---|---|---|
| **300** *(recommended)* | ~300 | Precise Q&A, definitions, specific facts |
| 700 | ~700 | Summaries, broad topics, fewer chunks |

### Embedding Models

| Model | Params | Dim | Avg Score | Speed |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` *(default)* | 22M | 384 | 16% | ⚡ 13 ms/query |
| `bge-base-en-v1.5` | 109M | 768 | **55%** | 🐢 54 ms/query |

> BGE scores ~3.4× higher on semantic similarity — use it when accuracy matters more than speed.

### Vector Stores

| Store | Persistence | Index Speed | Retrieval | Best For |
|---|---|---|---|---|
| **FAISS** *(default)* | RAM only | ⚡ Fastest | ⚡ 12–54 ms | All use cases |
| ChromaDB | Disk (local) | +400 ms overhead | 13–14 ms | Persistent collections |

### Language Models

| Model | Parameters | Approx RAM | Speed | Best For |
|---|---|---|---|---|
| **FLAN-T5-Base** *(recommended)* | 250M | ~1.2 GB | Fast | Most questions |
| FLAN-T5-Large | 780M | ~2.5 GB | Slower | Complex reasoning |

---

## 🔍 Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | venv not active / deps not installed | `pip install -r requirements.txt` |
| `"Answer not found"` | Query doesn't match any chunk | Rephrase or try 700-word chunks |
| Amber warning in chat tab | Embedding model changed after indexing | Click ⚡ Build Index again |
| Slow first query (~30s) | Model loading into RAM | Normal — subsequent queries are fast |
| `ChromaDB` error | Corrupt DB files | Delete `data/chroma_db/` and rebuild index |
| Out of memory | Large model + large docs | Use FLAN-T5-Base + FAISS + 300-word chunks |
| Port 8501 in use | Another Streamlit app running | `streamlit run app.py --server.port 8502` |
| Model download fails | No internet / firewall | Internet required on first run (~90–440 MB download) |
| Scanned PDF shows no results | Image-based PDF (no text layer) | Use text-based PDF or run OCR first |

---

## 🗂 Key Design Decisions

- **100% local** — No cloud APIs, no accounts, no data leaves your machine
- **Strict grounding** — FLAN-T5 is instructed to answer *only* from retrieved context, not training data
- **Extractive QA prompts** — Uses short, direct prompts that match FLAN-T5's training distribution; verbose role-play prompts cause hallucination loops
- **Per-chunk context budget** — 600 chars taken from each retrieved chunk (not first N chars of top chunk), so all chunks contribute context
- **Model mismatch guard** — Embedding model and vector store are locked together; changing one triggers a clear warning and blocks queries
- **Modular architecture** — Each concern (ingestion, chunking, embedding, storage, retrieval, generation) is isolated in its own module for easy testing and swapping
- **ChromaDB is file-based** — Like SQLite, stores data in `data/chroma_db/`. No separate server required

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ using **Streamlit · HuggingFace · FAISS · ChromaDB · PyMuPDF**

*100% offline · 100% private · Answers strictly grounded in your uploaded documents*

</div>
