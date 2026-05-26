# 🚀 Step-by-Step Setup Guide — University Course Assistant

## ✅ System Requirements
- Python 3.9, 3.10, or 3.11 (recommended: 3.10)
- RAM: Minimum 4GB (8GB recommended)
- Storage: ~2GB free (for model downloads)
- Internet: Required for first run (model download)

---

## 📦 STEP 1 — Install Python

Check your Python version:
```
python --version
```

If not installed, download from: https://www.python.org/downloads/

> ⚠️ Make sure Python is added to PATH during installation!

---

## 📁 STEP 2 — Extract the ZIP

Extract `rag_course_app.zip` to any folder.

You should see:
```
rag_course_app/
├── app.py
├── requirements.txt
├── README.md
├── SETUP_GUIDE.md
├── data/
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt.py
│   ├── generator.py
│   └── utils.py
└── experiments/
    └── evaluation.py
```

---

## 🐍 STEP 3 — Create Virtual Environment (Recommended)

Open terminal / command prompt inside the `rag_course_app` folder.

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` in your terminal — that means it's active ✅

---

## 📥 STEP 4 — Install Dependencies

```
pip install -r requirements.txt
```

This installs everything automatically. It may take 5–10 minutes.

> ⚠️ If you get errors on Windows with `faiss-cpu`, try:
> ```
> pip install faiss-cpu --no-cache-dir
> ```

> ⚠️ If `torch` install is slow, install it first separately:
> ```
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## ▶️ STEP 5 — Run the App

```
streamlit run app.py
```

The app opens in your browser at: **http://localhost:8501**

> 🌐 First time running: models (~100–440MB) will download from HuggingFace automatically.

---

## 🎓 STEP 6 — Use the App

1. Go to **📄 Upload Documents** tab
2. Upload your PDF files (syllabus, notes, assignments)
3. Click **⚡ Build Index**
4. Go to **💬 Ask Questions** tab
5. Ask anything about your documents!

---

## ❓ Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `torch` not found | `pip install torch` separately first |
| `chromadb` error | `pip install chromadb==0.4.24` |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Out of memory | Use FLAN-T5-Base + FAISS (not Large/ChromaDB) |
| Slow first query | Normal — model loads into RAM on first use |

---

## 💡 Tips

- Use **300-word chunks** for precise Q&A
- Use **FAISS** for fastest retrieval
- Use **FLAN-T5-Base** for fastest answers
- Enable **"Show Retrieved Chunks"** in sidebar to see what the AI is reading
- The AI will say *"Answer not found in uploaded documents"* if your question isn't covered

