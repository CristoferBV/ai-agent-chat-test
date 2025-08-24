# 🤖 Punta Blanca RAG Agent

Este proyecto implementa un **agente RAG (Retrieval-Augmented Generation)** usando **Google Gemini**, **LangChain**, **FAISS**, y **FastAPI**.  
Su propósito es responder preguntas sobre **Punta Blanca Solutions**, apoyándose en información extraída de su sitio web y de fuentes públicas.

---

## ✨ Características

- 🔎 **Retrieval**: búsqueda semántica sobre documentos indexados con FAISS.
- 📑 **Fuentes de conocimiento**:
  - Web scraping de [`puntablanca.ai`](https://www.puntablanca.ai/)
  - Archivos locales como `linkedin.md` (y opcionalmente `facts.md`).
- 🌐 **API REST con FastAPI** con documentación interactiva en `/docs`.
- 🧠 **Modelo generativo**: **Google Gemini 2.0 Flash** con respuestas estrictamente en JSON.
- 📦 **Contenerización con Docker**.
- ☁️ **Despliegue previsto en Google Cloud Run**.

---

## 📂 Estructura del proyecto

```
ai-agent-chat-test/
├── backend/
│   ├── app/                # Lógica de la API
│   │   ├── main.py         # Entrada FastAPI
│   │   ├── graph.py        # Flujo RAG con LangGraph
│   │   ├── retrieval.py    # Recuperación de contexto FAISS
│   │   ├── generation.py   # Conexión con Gemini
│   │   └── schemas.py      # Modelos Pydantic
│   │
│   ├── data/
│   │   ├── sources/        # Archivos locales (linkedin.md, facts.md, etc.)
│   │   └── vectorstore/    # Índice FAISS generado
│   │
│   ├── ingest/             # Scripts de ingestión/utilidades
│   │   ├── build_vectorstore.py
│   │   ├── inspect_vectorstore.py
│   │   └── search_probe.py
│   │
│   └── Dockerfile
│
├── docs/                   # Documentación extendida
├── requirements.txt
├── README.md
├── API.md
└── ARCHITECTURE.md
```

---

## 🚀 Ejecución local

### 1. Clonar el repositorio
```bash
git clone https://github.com/usuario/ai-agent-test.git
cd ai-agent-test
```

### 2. Instalar dependencias
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configurar `.env`
Crea `./backend/.env` (o en la raíz) con:

```env
GEMINI_API_KEY=tu_api_key_aqui
GEMINI_MODEL=gemini-2.0-flash
```

### 4. Construir el vectorstore
```bash
python backend/ingest/build_vectorstore.py
```

### 5. Levantar FastAPI
```bash
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8080
```
Abrir: http://localhost:8080/docs

---

## 📡 Ejemplo de consulta

```bash
curl -X POST "http://localhost:8080/api/ask"   -H "Content-Type: application/json"   -d '{"question": "¿Qué servicios ofrece Punta Blanca?"}'
```

**Respuesta esperada** (ejemplo):
```json
{
  "answer": "Punta Blanca ofrece soluciones de IA y estrategias de transformación digital...",
  "sources": [
    "https://www.puntablanca.ai/services",
    "file://linkedin.md"
  ],
  "confidence": 0.9
}
```

---

## 🐳 Contenerización

```bash
docker build -t ai-agent-chat-test .
docker run -p 8080:8080 ai-agent-chat-test
```

---

## ☁️ Despliegue en Cloud Run (resumen)

```bash
gcloud builds submit --tag gcr.io/<PROJECT_ID>/ai-agent
gcloud run deploy ai-agent   --image gcr.io/<PROJECT_ID>/ai-agent   --platform managed   --region us-central1   --allow-unauthenticated
```

---

## 👨‍💻 Autor

Desarrollado como **prueba técnica** para integrar **IA generativa + RAG** con despliegue en **Google Cloud**.
