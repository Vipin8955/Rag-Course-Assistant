<div align="center">

# 🎓 University Course Assistant

**A production-ready Retrieval-Augmented Generation (RAG) system for students.**  
Upload your course PDFs · Ask anything · Get answers strictly grounded in your documents.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📖 Table of Contents

- [What It Does](#-what-it-does)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration (.env)](#-configuration-env)
- [Usage Guide](#-usage-guide)
- [Configuration Options](#-configuration-options)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🤔 What It Does

The University Course Assistant lets students upload their **course PDFs** (lecture notes, syllabi, assignments) and ask natural language questions about them. The system retrieves the most relevant passages from the documents and generates answers **strictly grounded in what you uploaded** — no hallucinations, no external knowledge.

> **Think of it as a smart tutor that has read all your course material and can answer questions about it instantly.**

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INDEXING PIPELINE                           │
│                                                                     │
│  PDF Upload → Text Extraction → Word Chunking → Embedding Model     │
│                                                       ↓             │
│                                              Vector Index           │
│                                         (FAISS or ChromaDB)        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                             │
│                                                                     │
│  User Question → Embed Query → Semantic Search → Top-K Chunks       │
│                                                       ↓             │
│                                           Prompt Builder            │
│                                                  ↓                  │
│                                        FLAN-T5 LLM → Answer         │
└─────────────────────────────────────────────────────────────────────┘
```

1. **Ingest**: PDFs are parsed with PyMuPDF and cleaned.
2. **Chunk**: Text is split into overlapping word-based windows.
3. **Embed**: Each chunk is converted to a dense vector (384-dim) using a sentence transformer.
4. **Index**: Vectors are stored in FAISS (in-memory) or ChromaDB (persistent on disk).
5. **Retrieve**: At query time, the question is embedded and semantically matched against the index.
6. **Generate**: The top-K chunks are assembled into a prompt and passed to FLAN-T5, which generates a grounded answer.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 **PDF Ingestion** | PyMuPDF text extraction with cleaning (handles multi-page, multi-file) |
| ✂️ **Text Chunking** | Configurable 300 or 700 word chunks with 50–100 word overlap |
| 🧠 **Embedding Models** | MiniLM-L6-v2 (fast, 22M params) or BGE-base-en-v1.5 (accurate, 109M params) |
| 🗄️ **Vector Databases** | FAISS (in-memory) or ChromaDB (persistent local storage) |
| 🔍 **Semantic Retrieval** | L2-normalised cosine similarity, configurable Top-K |
| 💬 **Multi-turn Chat** | Chat UI with history, timestamps, and quick-prompt buttons |
| 🤖 **Local LLM** | FLAN-T5-Base or FLAN-T5-Large — runs 100% offline after first download |
| 📊 **Evaluation Tab** | Compare configurations on speed, chunk count, and retrieval score |
| ⚙️ **Config via .env** | All defaults controllable from a `.env` file — no code changes needed |
| 🛡️ **Error Handling** | Graceful errors at every layer with user-friendly messages |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | [Streamlit](https://streamlit.io) |
| **PDF Parsing** | [PyMuPDF](https://pymupdf.readthedocs.io) (`fitz`) |
| **Embeddings** | [sentence-transformers](https://www.sbert.net/) |
| **Vector DB (fast)** | [FAISS](https://github.com/facebookresearch/faiss) |
| **Vector DB (persistent)** | [ChromaDB](https://www.trychroma.com/) |
| **LLM** | [FLAN-T5](https://huggingface.co/google/flan-t5-base) via HuggingFace Transformers |
| **Charts** | [Plotly](https://plotly.com/python/) |
| **Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 📁 Project Structure

```
rag_course_app/
│
├── app.py                    # Main Streamlit application (UI + routing)
├── requirements.txt          # Pinned Python dependencies
├── .env                      # Local config overrides (gitignored)
├── .env.example              # Template — copy to .env and customise
├── README.md
├── SETUP_GUIDE.md            # Step-by-step beginner setup guide
│
├── src/                      # Core library (importable modules)
│   ├── config.py             # .env loader + centralised settings
│   ├── ingestion.py          # PDF save, extract, clean, load
│   ├── chunking.py           # Word-based text chunking with overlap
│   ├── embeddings.py         # Sentence embedding model wrappers
│   ├── vector_store.py       # FAISSStore + ChromaStore classes
│   ├── retriever.py          # Full index build + semantic retrieval
│   ├── prompt.py             # RAG prompt templates
│   ├── generator.py          # FLAN-T5 text generation pipeline
│   └── utils.py              # Shared helpers (formatting, hashing, etc.)
│
├── experiments/
│   └── evaluation.py         # Config comparison framework
│
└── data/                     # Runtime data (gitignored except .gitkeep)
    ├── .gitkeep              # Keeps data/ in git without contents
    └── chroma_db/            # ChromaDB persistent store (auto-created)
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9, 3.10, or 3.11** (recommended: 3.11)
- **4 GB RAM minimum** (8 GB recommended)
- **~2 GB free disk space** (for model downloads)
- **Internet connection** (required for first run only — to download models)

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

> ⏳ This installs ~15 packages including PyTorch and may take 5–10 minutes.

**Windows tip — if `torch` is slow:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)

```bash
cp .env.example .env
```

Edit `.env` to change default models, chunk sizes, or directories. The app works fine with defaults — this step is optional.

### 5. Run the App

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** 🎉

> 🌐 **First run**: Models (~100–440 MB) download automatically from HuggingFace. This happens once and is cached locally.

---

## 🔧 Configuration (.env)

Copy `.env.example` to `.env` and adjust:

```env
# Directories
DATA_DIR=data
CHROMA_DIR=data/chroma_db

# Default sidebar settings
DEFAULT_CHUNK_SIZE=300           # 300 or 700
DEFAULT_OVERLAP=75               # 50, 75, or 100
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2   # or bge-base-en-v1.5
DEFAULT_VECTOR_STORE=FAISS       # FAISS or ChromaDB
DEFAULT_LLM=FLAN-T5-Base (Fast)  # or FLAN-T5-Large
DEFAULT_TOP_K=5                  # 2–10

# Optional: HuggingFace token for private/gated models
# HF_TOKEN=hf_your_token_here

# Windows fixes (set automatically)
KMP_DUPLICATE_LIB_OK=TRUE
TOKENIZERS_PARALLELISM=false
```

> ⚠️ **Never commit your `.env` file.** It is listed in `.gitignore`. Use `.env.example` as the shared template.

---

## 📖 Usage Guide

### Step 1 — Upload Documents
1. Open the **📄 Upload Documents** tab
2. Drag & drop your PDFs (syllabus, lecture notes, assignments)
3. Click **⚡ Build Index** to process them

### Step 2 — Ask Questions
1. Switch to the **💬 Ask Questions** tab
2. Type your question or click a quick-prompt button
3. The assistant retrieves relevant chunks and generates a grounded answer

### Step 3 — Evaluate Configurations (Optional)
1. Go to the **📊 Evaluation** tab
2. Select combinations of embedding models, vector stores, and chunk sizes
3. Run evaluation to compare speed and retrieval quality

### Example Questions

| Question | What it tests |
|----------|--------------|
| `"Explain Unit 1"` | Broad summarisation |
| `"What is Machine Learning?"` | Definition retrieval |
| `"List the assignment requirements"` | Structured extraction |
| `"What are the learning objectives?"` | Syllabus-level retrieval |
| `"Summarise Chapter 2"` | Section-level summarisation |

---

## ⚙️ Configuration Options

### Chunk Size

| Size | Words | Best For |
|------|-------|---------|
| 300 | ~300 | Precise Q&A, definitions, specific facts |
| 700 | ~700 | Summaries, explanations, broad topics |

### Embedding Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `all-MiniLM-L6-v2` | 22M params | ⚡ Fast | Good |
| `bge-base-en-v1.5` | 109M params | 🐢 Slower | Better |

### Vector Stores

| Store | Storage | Speed | Best For |
|-------|---------|-------|---------|
| FAISS | RAM (lost on restart) | ⚡ Fastest | Quick experiments |
| ChromaDB | Disk (persistent) | Fast | Production, larger collections |

### Language Models

| Model | Parameters | RAM | Speed |
|-------|-----------|-----|-------|
| FLAN-T5-Base *(recommended)* | 250M | ~2 GB | Fast |
| FLAN-T5-Large | 780M | ~3 GB | Accurate |

---

## 🔍 Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError` | venv not active / deps not installed | `pip install -r requirements.txt` |
| `"Answer not found in uploaded documents"` | Query doesn't match any chunk | Rephrase the question or try larger chunk size (700) |
| Slow first query | Model loading into RAM | Normal — subsequent queries are fast |
| `ChromaDB` error | Corrupt DB files | Delete `data/chroma_db/` folder and rebuild the index |
| Out of memory | Large model + large docs | Use FLAN-T5-Base + FAISS + 300-word chunks |
| Port 8501 already in use | Another Streamlit app running | `streamlit run app.py --server.port 8502` |
| Model download fails | No internet / firewall | Connect to internet; first run downloads ~100–440 MB |
| Scanned PDF shows no results | Image-based PDF (no text layer) | Use a text-based PDF or run OCR first |
| `KMP_DUPLICATE_LIB_OK` warning | Windows OpenMP conflict | Already handled automatically — safe to ignore |

---

## 🗂 Key Design Decisions

- **100% local** — No cloud APIs, no accounts, no data leaves your machine.
- **Strict grounding** — The LLM is explicitly instructed to answer *only* from retrieved context, not from its training data.
- **Modular architecture** — Each concern (ingestion, chunking, embedding, storage, retrieval, generation) is isolated in its own module for easy testing and swapping.
- **ChromaDB is file-based** — Like SQLite, it stores data in a local folder (`data/chroma_db/`). No server required.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ using **Streamlit · HuggingFace · FAISS · ChromaDB**

*Answers strictly from your uploaded documents — no hallucination from external knowledge.*

</div>
