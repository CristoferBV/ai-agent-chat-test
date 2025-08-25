FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Avoid encoding problems
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUTF8=1

WORKDIR /app

# System dependencies for some libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the code
COPY backend /app

# ==== Warm-up cache (avoid downloads at runtime) ====
# 1) Location of the HF/transformers cache inside the image
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# 2) Download the sentence-transformers model during the build
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
PY

# =====================================================

# Port (Cloud Run uses $PORT; default 8080 is fine)
ENV PORT=8080
EXPOSE 8080

# Starting FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
