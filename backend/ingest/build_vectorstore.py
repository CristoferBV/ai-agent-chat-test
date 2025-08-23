"""
Script de ingesta para construir FAISS:
- Extrae texto limpio de https://www.puntablanca.ai/*
- Añade contenido público de LinkedIn pegado en backend/data/sources/linkedin.md
- Genera índice FAISS persistido en backend/data/vectorstore/faiss

Uso:
  python backend/ingest/build_vectorstore.py
"""
from __future__ import annotations
from pathlib import Path
import requests
import trafilatura
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE = Path(__file__).resolve().parents[1]     # .../backend
DATA_DIR = BASE / "data"
SOURCES_DIR = DATA_DIR / "sources"
VECTOR_DIR = DATA_DIR / "vectorstore" / "faiss"
SOURCES_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.parent.mkdir(parents=True, exist_ok=True)

SEED_URLS = [
    "https://www.puntablanca.ai/",
    "https://www.puntablanca.ai/services",
]

LOCAL_MARKDOWNS = [
    SOURCES_DIR / "linkedin.md",  
]

UA = {"User-Agent": "Mozilla/5.0"}

def fetch_clean(url: str) -> str:
    try:
        r = requests.get(url, timeout=25, headers=UA)
        r.raise_for_status()
        text = trafilatura.extract(r.text) or ""
        return text.strip()
    except Exception:
        return ""

def load_docs() -> list[Document]:
    docs: list[Document] = []
    for url in SEED_URLS:
        text = fetch_clean(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))
    for md in LOCAL_MARKDOWNS:
        if md.exists():
            docs.append(Document(page_content=md.read_text(encoding="utf-8"),
                                 metadata={"source": f"file://{md.name}"}))
    if not docs:
        raise RuntimeError("No se cargaron documentos. Revisa URLs y linkedin.md")
    return docs

def chunk_docs(docs: list[Document],
               chunk_size: int = 800,
               chunk_overlap: int = 120) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: list[Document] = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            out.append(Document(page_content=c, metadata=d.metadata))
    return out

def build_faiss(docs: list[Document]) -> None:
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embed)
    vs.save_local(str(VECTOR_DIR))
    print(f"✔ Vectorstore guardado en {VECTOR_DIR}")

if __name__ == "__main__":
    docs = load_docs()
    chunks = chunk_docs(docs)
    build_faiss(chunks)
