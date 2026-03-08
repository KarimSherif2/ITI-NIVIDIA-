"""
ingest.py
Document ingestion pipeline - loads PDF/DOCX, chunks, embeds, saves to FAISS.
PDF parsing priority: PyMuPDF -> pdfplumber -> pypdf (fallback)
"""

import os

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_PATH = r"D:\WORKSPACE\RAG_PROJECT\vectorstore"


def load_pdf(file_path: str) -> list:
    """Try PyMuPDF first, then pdfplumber, then fallback to pypdf."""
    if PYMUPDF_AVAILABLE:
        print("  Using PyMuPDF parser...")
        docs = []
        pdf = fitz.open(file_path)
        for i, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path, "page": i}
                ))
        pdf.close()
        if docs:
            return docs

    if PDFPLUMBER_AVAILABLE:
        print("  Using pdfplumber parser...")
        docs = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": i}
                    ))
        if docs:
            return docs

    print("  Using pypdf fallback parser...")
    loader = PyPDFLoader(file_path)
    return loader.load()


def ingest_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        print("Loading PDF document...")
        docs = load_pdf(file_path)
    elif ext == ".docx":
        print("Loading DOCX document...")
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .pdf and .docx are supported.")

    if not docs:
        raise ValueError("The document appears to be empty or could not be parsed.")

    print(f"Loaded {len(docs)} page(s). Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = FAISS.from_documents(chunks, embeddings)

    print("Saving vectorstore...")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectordb.save_local(VECTORSTORE_PATH)
    print("Vectorstore saved successfully!")

    return vectordb


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "contract.pdf"
    ingest_document(path)