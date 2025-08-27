# AI Agent RAG · FastAPI + FAISS + Gemini (Cloud Run)

Este repositorio implementa un **agente de preguntas y respuestas (RAG)** para consultar conocimiento propio (archivos `.md`) usando *embeddings* y un *vector store* local (**FAISS**). La API está construida con **FastAPI**, orquestada con **LangGraph/LangChain**, y utiliza **Gemini** como modelo generativo. El servicio se empaqueta con **Docker**, se construye con **Cloud Build** y se despliega sin servidores en **Cloud Run**.

> Estado actual: probado localmente y desplegado en Cloud Run con éxito.  
> URL de ejemplo (la tuya será distinta): `https://ai-agent-1056962430201.us-central1.run.app/`

---

## Tabla de contenidos
- [Arquitectura](#arquitectura)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Variables de entorno](#variables-de-entorno)
- [Uso local](#uso-local)
- [Ingesta y construcción del vector store](#ingesta-y-construcción-del-vector-store)
- [Build & Deploy en Google Cloud](#build--deploy-en-google-cloud)
  - [1) Preparación del proyecto](#1-preparación-del-proyecto)
  - [2) Build con Cloud Build](#2-build-con-cloud-build)
  - [3) Deploy en Cloud Run](#3-deploy-en-cloud-run)
  - [4) Verificación](#4-verificación)
- [Endpoints](#endpoints)
- [Solución de problemas](#solución-de-problemas)
- [Notas de seguridad](#notas-de-seguridad)
- [Licencia](#licencia)

---

## Arquitectura

```
+---------------------+          +-----------------+
|   Cliente / UI      |  HTTPS   |   Cloud Run     |
| (curl, web, React)  +--------->+  FastAPI (API)  |
+---------------------+          |  /api/ask       |
                                 |  /docs          |
                                 +--------+--------+
                                          |
                                          | LangGraph/LangChain
                                          v
                                 +--------+--------+
                                 |  Retrieval (FAISS)
                                 |  /app/data/vectorstore/faiss
                                 +--------+--------+
                                          |
                                          | sentence-transformers
                                          v
                                 +-----------------+
                                 | Gemini (LLM)    |
                                 +-----------------+
```

- **RAG**: recupera fragmentos relevantes desde FAISS y el LLM (Gemini) redacta la respuesta.
- **FAISS**: índice local (archivos `index.faiss` y `index.pkl`).
- **Ingesta**: `ingest/build_vectorstore.py` lee `.md` en `data/sources/`, crea embeddings y guarda el índice.
- **API**: `FastAPI` expone `/api/ask` y documentación `/docs` (Swagger).
- **Infra**: `Dockerfile` + `cloudbuild.yaml` → imagen en Artifact Registry → Cloud Run.

---

## Estructura del proyecto

```
backend/
├─ app/
│  ├─ generation.py      # Inicializa Gemini
│  ├─ graph.py           # Construye el grafo LangGraph (RAG pipeline)
│  ├─ main.py            # FastAPI: rutas /, /api/ask
│  ├─ retrieval.py       # Carga FAISS y ejecuta búsquedas de vectores
│  └─ schemas.py         # Esquemas Pydantic: AskRequest/AskResponse
│
├─ data/
│  ├─ sources/           # Tu conocimiento fuente (.md y links)
│  │  ├─ facts.md
│  │  └─ linkedin.md
│  └─ vectorstore/faiss/ # Índice FAISS generado
│     ├─ index.faiss
│     └─ index.pkl
│
├─ ingest/
│  └─ build_vectorstore.py  # Script para construir el índice FAISS
│
├─ Dockerfile
├─ requirements.txt
├─ .dockerignore
└─ cloudbuild.yaml
```

> Importante: **no** subas claves ni `.env` al repo. La API key de Gemini se pasa como variable de entorno o como secreto.

---

## Requisitos

- **Python 3.11+** (si ejecutarás localmente sin Docker)
- **Docker** (para local y para build reproducible)
- **Cuenta de Google Cloud** con:
  - `Cloud Build`, `Artifact Registry` y `Cloud Run` habilitados
  - `gcloud` CLI configurado
- **Gemini API Key** (Google AI Studio)

---

## Variables de entorno

La API usa estas variables:
- `GEMINI_API_KEY`: clave de Gemini.
- `GEMINI_MODEL`: por defecto `gemini-2.0-flash` (puedes usar otro modelo disponible).

> En producción (Cloud Run) **NO** copies `.env` dentro de la imagen. Define estas variables vía `--set-env-vars` o usa **Secret Manager**.

---

## Uso local

### 1) Crear y activar venv
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows PowerShell
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 2) Construir el vector store (una sola vez o cuando cambie `data/sources/`)
```bash
python backend/ingest/build_vectorstore.py
```
Genera `backend/data/vectorstore/faiss/index.faiss` y `index.pkl`.

### 3) Ejecutar la API
```bash
export GEMINI_API_KEY="TU_CLAVE"
export GEMINI_MODEL="gemini-2.0-flash"
uvicorn backend.app.main:app --reload --port 8080
```
Abre `http://localhost:8080/docs`.

---

## Ingesta y construcción del vector store

Coloca tus fuentes en **`backend/data/sources/`** (por ejemplo `.md`, `.txt`, etc., según soporte del script).  
Ejecuta:

```bash
python backend/ingest/build_vectorstore.py
```

Este script:
1. Lee y parte el texto en *chunks* (`langchain-text-splitters`).
2. Genera embeddings con **sentence-transformers** (modelo: `paraphrase-multilingual-MiniLM-L12-v2`).
3. Crea/actualiza el índice **FAISS** en `backend/data/vectorstore/faiss/`.

> Si cambias los documentos fuente, **re-ejecuta** este script y vuelve a desplegar si el índice viaja dentro de la imagen.

---

## Build & Deploy en Google Cloud

> A continuación se describe el flujo reproducido en este proyecto.

### 1) Preparación del proyecto

Selecciona proyecto y región:
```bash
gcloud config set project <PROJECT_ID>
gcloud config set run/region us-central1
```

Habilita APIs:
```bash
gcloud services enable artifactregistry.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com
```

Crea el repositorio de imágenes (una única vez):
```bash
gcloud artifacts repositories create containers \
  --repository-format=docker \
  --location=us-central1 \
  --description="Repo de imágenes"
```

### 2) Build con Cloud Build (Crea el archivo 'cloudbuild.yaml' en la raiz del repositorio)

El `cloudbuild.yaml` simple etiqueta la imagen como `:latest`:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'us-central1-docker.pkg.dev/$PROJECT_ID/containers/ai-agent:latest'
      - '-f'
      - 'backend/Dockerfile'
      - '.'

images:
  - 'us-central1-docker.pkg.dev/$PROJECT_ID/containers/ai-agent:latest'
```

Ejecuta el build desde la **raíz** del repo:
```bash
gcloud builds submit --config=cloudbuild.yaml --project=<PROJECT_ID>
```

### 3) Deploy en Cloud Run

```bash
gcloud run deploy ai-agent \
  --image us-central1-docker.pkg.dev/<PROJECT_ID>/containers/ai-agent:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars GEMINI_MODEL=gemini-2.0-flash \
  --set-env-vars GEMINI_API_KEY=TU_CLAVE_GEMINI
```

> Recomendado (más seguro): usar **Secret Manager** y `--update-secrets=GEMINI_API_KEY=gemini_api_key:latest`.

### 4) Verificación

- Raíz: `GET /` → `{"service":"ai-agent","status":"ok","docs":"/docs"}`
- Documentación: `GET /docs`

Logs en vivo:
```bash
gcloud run services logs tail ai-agent --region us-central1
```

---

## Endpoints

- `GET /`  
  Respuesta simple con estado y enlace a `/docs`.

- `POST /api/ask`  
  **Request (JSON)**
  ```json
  { "question": "¿Qué es Punta Blanca Solutions?" }
  ```
  **Response (JSON)**
  ```json
  {
    "answer": "…",
    "sources": ["facts.md#L20-L40", "linkedin.md#..."],
    "confidence": 0.73
  }
  ```

- `GET /docs`  
  UI interactiva de Swagger.

> Los modelos de entrada/salida se definen en `backend/app/schemas.py` (Pydantic).

---

## Notas de seguridad

- **Nunca** publiques tu `GEMINI_API_KEY` en el repo o en comandos compartidos.
- Usa **Secret Manager** en GCP para inyectar secretos.
- Revisa CORS y limita `allow_origins` si expones esta API a un frontend específico.
- No subas documentos sensibles a `data/sources/` si la imagen de Docker será pública.

---

## Licencia

MIT — ver `LICENSE` si aplica. Puedes adaptar y reutilizar este proyecto en tus propios despliegues.
