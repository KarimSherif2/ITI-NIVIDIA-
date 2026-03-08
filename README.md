# 📄 Smart Contract Q&A Assistant

An end-to-end RAG application to upload legal contracts (PDF/DOCX) and interact with them via chat — powered by LangChain, FAISS, Groq (llama-3.3-70b), and Gradio.

---

## 🗂️ Project Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio UI — Upload and Chat tabs |
| `ingest.py` | Document loading, chunking, embedding |
| `rag_pipeline.py` | FAISS retrieval + Groq LLM |
| `guardrails.py` | Input safety + output grounding |
| `summarize.py` | Structured contract summarization |
| `evaluate.py` | Evaluation pipeline with metrics |
| `utils.py` | Source formatting helpers |
| `server.py` | FastAPI + LangServe REST API |
| `requirements.txt` | Python dependencies |
| `Evaluation_Report.docx` | Full written evaluation report |

---

## ⚙️ Setup

### 1. Get a Groq API Key (free)
Sign up at https://console.groq.com → API Keys → Create new key.
No credit card required.

### 2. Add your key to `rag_pipeline.py`
Open `rag_pipeline.py` and set your key on line 16:
```python
GROQ_API_KEY = "your-groq-key-here"
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running

### Gradio UI
```bash
python app.py
```
Open http://localhost:7860

**Workflow:**
1. **Upload Contract tab** → select PDF or DOCX → click Process Document
2. **Chat tab** → ask questions → click Ask (or press Enter)
3. Click **Summarize Contract** for a structured 7-section overview

### API Server (optional)
```bash
uvicorn server:app --reload --port 8000
```
- Docs: http://localhost:8000/docs
- Call: `POST http://localhost:8000/rag/invoke` with `{"input": {"query": "..."}}`

### Evaluation
Upload a document first, then:
```bash
python evaluate.py
```
Results saved to `evaluation_report.json`

---

## 📊 Evaluation Results (Form 1099-MISC)

| Metric | Result | Target |
|--------|--------|--------|
| Pass Rate | **100%** (10/10) | >80% |
| Avg Latency | **0.44s** | <5s |
| Grounded Rate | **100%** | 100% |
| Avg Keyword Hit Rate | **0.90** | >0.5 |

---

## 🛡️ Guardrails

| Layer | What it does |
|-------|-------------|
| Input safety | Blocks harmful or malicious questions |
| Output grounding | Flags answers with no supporting source chunks |
| LLM honesty | Returns "I could not find the answer" for off-topic questions |

---

## 🔧 Configuration

| Setting | Where | Default | Options |
|---------|-------|---------|---------|
| LLM model | `rag_pipeline.py` | `llama-3.3-70b-versatile` | Any Groq model |
| Vector store | `rag_pipeline.py` | `faiss` | `chroma` (install chromadb) |
| Embeddings | `rag_pipeline.py` | `sentence_transformers` | `openai` (install langchain-openai) |
| Chunk size | `ingest.py` | 1000 | Increase for longer context |
| Retrieved chunks | `rag_pipeline.py` | 5 | Reduce to 3 for faster responses |

---

## 🚨 Troubleshooting

| Problem | Fix |
|---------|-----|
| `model_decommissioned` error | Update model name in `rag_pipeline.py` — check https://console.groq.com/docs/models |
| `vectorstore not found` | Upload and process a document first |
| DOCX not loading | `pip install docx2txt` |
| PDF not loading | `pip install pymupdf pdfplumber` |
| Groq API error 401 | Check your API key is set correctly in `rag_pipeline.py` |

---

## ⚠️ Limitations

- English documents only
- Single document per session
- No persistent memory across app restarts (use Save Chat History button)
- Relevance score (0.33) is lower than target due to MiniLM embedding model on short factual answers — answers are correct despite this

---

## 🔮 Future Enhancements

- Multi-document search
- Domain-specific fine-tuned models
- Role-based access control
- Cloud deployment (Docker/Kubernetes)
- Persistent memory across sessions
- Language detection for non-English documents