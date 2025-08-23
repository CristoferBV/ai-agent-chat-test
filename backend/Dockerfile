FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (some libs require it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the code
COPY backend /app

# (Optional) Pre-download the embeddings model in build to speed up
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✔ Embedding model cached")
PY

# --- Automate ingestion in the build ---
# You can parameterize which URLs to scrape and whether you want to skip it:
# docker build --build-arg SCRAPE_URLS="https://www.puntablanca.ai/,https://www.puntablanca.ai/services" .
ARG SCRAPE_URLS="https://www.puntablanca.ai/"
ARG SKIP_SCRAPE=0
ENV SCRAPE_URLS=$SCRAPE_URLS
ENV SKIP_SCRAPE=$SKIP_SCRAPE

RUN python ingest/build_vectorstore.py

# Cloud Run Port
ENV PORT=8080
EXPOSE 8080

# API Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
