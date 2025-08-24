from __future__ import annotations
from pathlib import Path
import requests
import trafilatura
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
SOURCES_DIR = DATA_DIR / "sources"
VECTOR_DIR = DATA_DIR / "vectorstore" / "faiss"
SOURCES_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.parent.mkdir(parents=True, exist_ok=True)

SEED_URLS = [
    "https://www.puntablanca.ai/", 
    "https://www.puntablanca.ai/services",
    "https://www.puntablanca.ai/about-us",
    "https://www.puntablanca.ai/contact-us",
]

LOCAL_MARKDOWNS = [
    SOURCES_DIR / "linkedin.md",
    SOURCES_DIR / "facts.md",
]

UA = {"User-Agent": "Mozilla/5.0"}

def fetch_clean(url: str) -> str:
    try:
        r = requests.get(url, timeout=25, headers=UA)
        r.raise_for_status()
       
        return (trafilatura.extract(
            r.text,
            url=url,
            favor_recall=True,
            include_links=True,
            include_tables=True
        ) or "").strip()
    except Exception:
        return ""

def load_docs() -> list[Document]:
    docs: list[Document] = []
    # Web
    for url in SEED_URLS:
        text = fetch_clean(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))
    # Local (copied public LinkedIn)
    for md in LOCAL_MARKDOWNS:
        if md.exists():
            txt = md.read_text(encoding="utf-8").strip()
            if txt:
                docs.append(Document(page_content=txt, metadata={"source": f"file://{md.name}"}))
    if not docs:
        raise RuntimeError("No se cargaron documentos. Revisa URLs y que linkedin.md tenga texto.")
    return docs

def chunk_docs(docs: list[Document], chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for d in docs:
        for c in splitter.split_text(d.page_content):
            out.append(Document(page_content=c, metadata=d.metadata))
    return out

def build_faiss(docs: list[Document]) -> None:
    # MULTILINGUAL (same as in retrieval.py) embedding
    embed = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    vs = FAISS.from_documents(docs, embed)
    vs.save_local(str(VECTOR_DIR))
    print(f"Vectorstore guardado en {VECTOR_DIR}")

if __name__ == "__main__":
    docs = load_docs()
    chunks = chunk_docs(docs)
    build_faiss(chunks)
