# backend/Dockerfile
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUTF8=1

WORKDIR /app

# ---- System deps (faiss, sentence-transformers y build wheels) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Primero requirements para aprovechar cache ----
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copia del código ----
COPY backend /app

# ---- Cache de modelos dentro de la imagen (opcional) ----
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Pre-descargar el modelo para evitar downloads en cold start (puedes quitarlo si prefieres)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# ---- Cloud Run usa $PORT ----
ENV PORT=8080
EXPOSE 8080

# ---- Usuario no-root (recomendado) ----
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- Arranque respetando $PORT ----
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
