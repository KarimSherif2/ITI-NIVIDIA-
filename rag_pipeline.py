"""
rag_pipeline.py
RAG pipeline - FAISS retrieval + Groq LLM (free, fast).

Set your Groq API key:
  Windows: $env:GROQ_API_KEY = "your-key-here"
  Or hardcode it in the GROQ_API_KEY variable below.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_transformers import LongContextReorder

VECTORSTORE_PATH = "vectorstore"
GROQ_API_KEY     = "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"


# ── Embeddings ─────────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ── LLM (Groq - free & fast) ───────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY
)


# ── Prompt ─────────────────────────────────────────────────────────────────────

prompt = ChatPromptTemplate.from_template("""You are a legal contract assistant.

DISCLAIMER: This tool provides informational assistance only and does NOT
constitute legal advice. Always consult a qualified legal professional.
This system only processes English-language documents.

Use ONLY the provided context below to answer the question.
Do NOT use any prior knowledge outside the document.
If the answer is not in the context, say exactly:
"I could not find the answer in the contract."

Context:
{context}

Conversation so far:
{history}

Question:
{question}

Answer:""")

reorder = LongContextReorder()


# ── Vectorstore (lazy loaded) ──────────────────────────────────────────────────

_docstore  = None
_retriever = None


def _load_faiss():
    global _docstore
    from langchain_community.vectorstores import FAISS
    print("Loading FAISS vectorstore...")
    _docstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def _load_vectorstore():
    global _docstore, _retriever
    _load_faiss()
    _retriever = _docstore.as_retriever(search_kwargs={"k": 5})
    return _docstore, _retriever


def get_docstore():
    if _docstore is None:
        return _load_vectorstore()[0]
    return _docstore


def get_retriever():
    if _retriever is None:
        return _load_vectorstore()[1]
    return _retriever


def reload_vectorstore():
    return _load_vectorstore()


# ── RAG chain ──────────────────────────────────────────────────────────────────

def rag_chain(inputs: dict) -> dict:
    question = inputs["query"]
    history  = inputs.get("history", [])

    retriever = get_retriever()
    docs = retriever.invoke(question)
    docs = reorder.transform_documents(docs)
    context = "\n\n".join([d.page_content for d in docs])

    history_text = "None yet."
    if history:
        lines = []
        for u, b in history[-3:]:
            if u:
                lines.append(f"User: {u}")
            if b:
                clean_b = b.split("\n\n📄 Sources:")[0]
                lines.append(f"Assistant: {clean_b}")
        if lines:
            history_text = "\n".join(lines)

    final_prompt = prompt.invoke({
        "context":  context,
        "history":  history_text,
        "question": question
    })

    answer = llm.invoke(final_prompt).content

    return {"result": answer, "source_documents": docs}


def load_rag():
    try:
        reload_vectorstore()
    except Exception:
        pass
    return rag_chain