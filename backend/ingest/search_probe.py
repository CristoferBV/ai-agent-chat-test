# backend/ingest/search_probe.py
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(__file__).resolve().parents[1]
VECTOR_DIR = BASE / "data" / "vectorstore" / "faiss"

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vs = FAISS.load_local(str(VECTOR_DIR), embed, allow_dangerous_deserialization=True)

q = "CEO Punta Blanca"
docs = vs.similarity_search(q, k=5)
print(f"Query: {q}\n")
for i, d in enumerate(docs, 1):
    print(f"{i}. {d.metadata.get('source')}")
    print("   ", d.page_content[:180].replace("\n", " "), "...\n")
