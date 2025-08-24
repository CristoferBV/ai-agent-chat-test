# 📡 API Documentation – Punta Blanca RAG Agent

La API está construida en **FastAPI** y expone un endpoint principal para consultas RAG.

---

## 🔎 Endpoints

### `GET /healthz`
Verificación de salud del servicio.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /api/ask`

Permite enviar una pregunta al agente RAG.

#### Request Body
```json
{
  "question": "¿Qué servicios ofrece Punta Blanca?"
}
```

#### Response (200 OK)
```json
{
  "answer": "Punta Blanca ofrece soluciones de IA...",
  "sources": [
    "https://www.puntablanca.ai/services",
    "file://linkedin.md"
  ],
  "confidence": 0.9
}
```

#### Response (422 Validation Error)
```json
{
  "detail": [
    {
      "loc": ["body", "question"],
      "msg": "Pregunta demasiado corta",
      "type": "value_error"
    }
  ]
}
```

---

## 📑 Modelos

### `AskRequest`
- **question** *(string, min_length=3)* → Pregunta del usuario

### `AskResponse`
- **answer** *(string)* → Respuesta generada  
- **sources** *(string[])* → Fuentes usadas  
- **confidence** *(float)* → Confianza 0..1

---

## 🌐 Swagger UI

Disponible en:  
👉 http://localhost:8080/docs
