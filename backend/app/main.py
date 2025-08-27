import warnings
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.responses import JSONResponse, RedirectResponse

from .schemas import AskRequest, AskResponse
from .retrieval import Retriever
from .generation import init_gemini
from .graph import build_graph

from dotenv import load_dotenv

# ---------- .env ----------
for p in [
    Path(__file__).resolve().parents[1] / ".env",  # backend/.env
    Path(__file__).resolve().parents[2] / ".env",  # repo/.env
    Path.cwd() / ".env",                           # cwd/.env
]:
    if p.exists():
        load_dotenv(p, override=False)

# ---------- Logging / Warnings ----------
logging.getLogger("gen").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- FastAPI ----------
# ORJSONResponse -> returns UTF-8 without escaping accents/ñ
app = FastAPI(
    title="Punta Blanca RAG Agent",
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Resources ----------
retriever = Retriever()
gemini = init_gemini()
graph = build_graph(retriever, gemini)

# ---------- Endpoints ----------

@app.get("/", include_in_schema=False)
def root():
    return {"service": "ai-agent", "status": "ok", "docs": "/docs"}

@app.get("/healthz", include_in_schema=False)
def healthz():
    return JSONResponse({"ok": True})

@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(payload: AskRequest):
    try:
        out = graph.invoke({"question": payload.question})

        # Normalize structure
        result = out
        if isinstance(out, dict) and "result" in out and isinstance(out["result"], dict):
            result = out["result"]
        if isinstance(result, list) and result and isinstance(result[-1], dict):
            result = result[-1]

        answer = str(result.get("answer", "")).strip()
        sources = list(result.get("sources", []))
        confidence = float(result.get("confidence", 0.5))

        return ORJSONResponse(
            content={"answer": answer, "sources": sources, "confidence": confidence}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
