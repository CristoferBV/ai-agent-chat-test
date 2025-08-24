from pathlib import Path
from collections import Counter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE = Path(__file__).resolve().parents[1]
VECTOR_DIR = BASE / "data" / "vectorstore" / "faiss"

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vs = FAISS.load_local(str(VECTOR_DIR), embed, allow_dangerous_deserialization=True)

# Recuperar muchos documentos
docs = vs.similarity_search("dummy", k=1000)

# Contar fuentes
sources = [d.metadata.get("source", "unknown") for d in docs]
counts = Counter(sources)

print("📊 Documentos por fuente:\n")
for src, cnt in counts.items():
    print(f"- {src}: {cnt} chunks")
