"""
server.py
FastAPI + LangServe microservice exposing the RAG pipeline.

Run with:
    uvicorn server:app --reload --port 8000

Then call:
    POST http://localhost:8000/rag/invoke
    Body: {"input": {"query": "What is the purpose of this contract?"}}

API docs available at: http://localhost:8000/docs
"""

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from rag_pipeline import rag_chain

app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG-based contract Q&A via LangServe",
    version="1.0.0"
)

runnable_rag = RunnableLambda(rag_chain)

add_routes(
    app,
    runnable_rag,
    path="/rag",
    input_type=dict,
    output_type=dict,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
