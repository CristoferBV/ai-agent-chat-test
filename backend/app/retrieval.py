from pathlib import Path
from typing import Tuple, List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class Retriever:
    """
    Carga el índice FAISS (generado por ingest/build_vectorstore.py)
    y expone get_context() para recuperar contexto + fuentes.
    """
    def __init__(self, faiss_subpath: str | None = None):
        base = Path(__file__).resolve().parents[1]  # .../backend
        faiss_dir = Path(faiss_subpath) if faiss_subpath else base / "data" / "vectorstore" / "faiss"
        if not faiss_dir.exists():
            raise RuntimeError(f"No se encontró el índice FAISS en: {faiss_dir}")

        # Embedding MULTILINGÜE (ES/EN) para que preguntas en español funcionen con docs en inglés
        self.embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # allow_dangerous_deserialization requerido por FAISS.load_local
        self.vs: FAISS = FAISS.load_local(str(faiss_dir), self.embed, allow_dangerous_deserialization=True)

    def get_context(self, question: str, k: int = 4) -> Tuple[str, List[str], list[Document]]:
        docs = self.vs.similarity_search(question, k=k)
        context = "\n\n".join([d.page_content for d in docs])
        sources = list({d.metadata.get("source", "unknown") for d in docs})
        # Debug pequeño (opcional):
        # print(f"[retrieval] chars={len(context)} sources={sources[:3]}")
        return context, sources, docs
