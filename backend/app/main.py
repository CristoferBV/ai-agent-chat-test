from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import AskRequest, AskResponse
from .retrieval import Retriever
from .generation import init_gemini
from .graph import build_graph

# --- Cargar .env (python-dotenv) ---
from dotenv import load_dotenv
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

for p in [
    Path(__file__).resolve().parents[1] / ".env",  # backend/.env
    Path(__file__).resolve().parents[2] / ".env",  # repo/.env (opcional)
    Path.cwd() / ".env",                           # cwd/.env
]:
    if p.exists():
        load_dotenv(p, override=False)

app = FastAPI(title="Punta Blanca RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# Inicializa recursos al arrancar
retriever = Retriever()
gemini = init_gemini()
graph = build_graph(retriever, gemini)

@app.get("/healthz", tags=["health"])
def healthz():
    return {"status": "ok"}

@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(payload: AskRequest):
    try:
        result = graph.invoke({"question": payload.question})
        answer = str(result.get("answer", "")).strip()
        sources = list(result.get("sources", []))
        confidence = float(result.get("confidence", 0.5))
        return AskResponse(answer=answer, sources=sources, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
