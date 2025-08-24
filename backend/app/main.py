import logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import AskRequest, AskResponse
from .retrieval import Retriever
from .generation import init_gemini
from .graph import build_graph

from dotenv import load_dotenv
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)

for p in [
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parents[2] / ".env",
    Path.cwd() / ".env",
]:
    if p.exists():
        load_dotenv(p, override=False)

app = FastAPI(title="Punta Blanca RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# Initialize resources at startup
retriever = Retriever()
gemini = init_gemini()
graph = build_graph(retriever, gemini)

@app.get("/healthz", tags=["health"])
def healthz():
    return {"status": "ok"}

@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(payload: AskRequest):
    try:
        out = graph.invoke({"question": payload.question})

        # Optional Debug
        print(">>> graph output:", out)

        # Ensure correct shape
        result = out
        if isinstance(out, dict) and "result" in out and isinstance(out["result"], dict):
            result = out["result"]

        # Another case that sometimes occurs: list with a single final state
        if isinstance(result, list) and result and isinstance(result[-1], dict):
            result = result[-1]

        answer = str(result.get("answer", "")).strip()
        sources = list(result.get("sources", []))
        confidence = float(result.get("confidence", 0.5))

        return AskResponse(answer=answer, sources=sources, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

