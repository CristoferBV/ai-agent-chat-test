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

# ---- First requirements to take advantage of cache ----
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy code ----
COPY backend /app

# ---- Cache de modelos dentro de la imagen (opcional) ----
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Pre-download the model to avoid cold start downloads
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# ---- Cloud Run uses $PORT ----
ENV PORT=8080
EXPOSE 8080

# ---- Non-root user (recommended) ----
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ---- Start respecting $PORT ----
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
