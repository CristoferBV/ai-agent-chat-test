from __future__ import annotations
from pathlib import Path
import os
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

# --- Environment Configuration ---
# CSV of URLs, by default only the home (secure)
urls_env = os.getenv("SCRAPE_URLS", "https://www.puntablanca.ai/")
SEED_URLS = [u.strip() for u in urls_env.split(",") if u.strip()]

# If SKIP_SCRAPE=1, it does not make requests; only uses local files (linkedin.md)
SKIP_SCRAPE = os.getenv("SKIP_SCRAPE", "0") == "1"

LOCAL_MARKDOWNS = [
    SOURCES_DIR / "linkedin.md",  # place public content here
]

UA = {"User-Agent": "Mozilla/5.0"}

def fetch_clean(url: str) -> str:
    if SKIP_SCRAPE:
        return ""
    try:
        r = requests.get(url, timeout=25, headers=UA)
        r.raise_for_status()
        text = trafilatura.extract(r.text) or ""
        return text.strip()
    except Exception:
        return ""

def load_docs() -> list[Document]:
    docs: list[Document] = []
    # Web
    for url in SEED_URLS:
        text = fetch_clean(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url}))
    # Local (LinkedIn public copied)
    for md in LOCAL_MARKDOWNS:
        if md.exists():
            txt = md.read_text(encoding="utf-8").strip()
            if txt:
                docs.append(Document(page_content=txt, metadata={"source": f"file://{md.name}"}))
    if not docs:
        raise RuntimeError("No se cargaron documentos. Revisa SKIP_SCRAPE/URLs y que linkedin.md tenga texto.")
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
    print(f"Vectorstore guardado en {VECTOR_DIR}")

if __name__ == "__main__":
    docs = load_docs()
    chunks = chunk_docs(docs)
    build_faiss(chunks)
