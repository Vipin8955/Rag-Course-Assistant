# 🎓 University Course Assistant

A complete **Retrieval-Augmented Generation (RAG)** system that lets students upload course PDFs and ask questions — with answers grounded strictly in their uploaded documents.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 PDF Ingestion | PyMuPDF text extraction with cleaning |
| ✂️ Text Chunking | 300 or 700 words, 50–100 word overlap |
| 🧠 Embeddings | MiniLM-L6-v2 or BGE-base-en-v1.5 |
| 🗄️ Vector DB | FAISS (in-memory) or ChromaDB (persistent) |
| 🔍 Retrieval | Semantic search, configurable Top-K |
| 💬 Chat UI | Multi-turn chat with history |
| 🤖 LLM | FLAN-T5 (Base or Large) |
| 📊 Evaluation | Compare configs on speed & accuracy |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download model weights from HuggingFace (~100MB for MiniLM, ~440MB for BGE).

### 2. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
rag_course_app/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Uploaded PDFs + ChromaDB storage
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # PDF upload & text extraction
│   ├── chunking.py         # Word-based text chunking
│   ├── embeddings.py       # Sentence embedding models
│   ├── vector_store.py     # FAISS & ChromaDB stores
│   ├── retriever.py        # Full index build + retrieval
│   ├── prompt.py           # Prompt templates
│   ├── generator.py        # LLM text generation
│   └── utils.py            # Shared helpers
│
└── experiments/
    ├── __init__.py
    └── evaluation.py       # Config comparison framework
```

---

## 🛠️ Configuration Options

### Sidebar Settings

| Setting | Options | Default |
|---------|---------|---------|
| Chunk Size | 300, 700 words | 300 |
| Overlap | 50–100 words | 75 |
| Embedding Model | MiniLM, BGE | MiniLM |
| Vector Store | FAISS, ChromaDB | FAISS |
| LLM | FLAN-T5-Base, Large | Base |
| Top-K | 2–10 | 5 |

---

## 📊 How It Works

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Index
                                                              ↓
User Query → Query Embedding → Semantic Search → Top-K Chunks
                                                              ↓
                                          Prompt Builder → LLM → Answer
```

---

## 💡 Example Questions

- *"Explain Unit 1"*
- *"What is Machine Learning?"*
- *"Summarise Chapter 2"*
- *"What are the assignment requirements?"*
- *"List the learning objectives"*

---

## ⚠️ Notes

- The LLM answers **ONLY** from uploaded documents
- First run downloads model weights (requires internet)
- FLAN-T5-Base requires ~2GB RAM
- ChromaDB stores data persistently in `data/chroma_db/`

---

## 📦 Dependencies

- **Streamlit** — Web UI framework
- **PyMuPDF (fitz)** — PDF text extraction
- **sentence-transformers** — Embedding models
- **faiss-cpu** — Vector similarity search
- **chromadb** — Persistent vector database
- **transformers** — HuggingFace LLM
- **plotly** — Evaluation charts
- **pandas** — Data manipulation
