import logging
import os, json
import google.generativeai as genai

MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.0-flash"

# Prompt del sistema (instrucciones globales)
SYSTEM_PROMPT = (
    "Eres un asistente que responde únicamente con la información del 'Contexto'. "
    "Si no hay evidencia suficiente, responde: "
    "\"No hay información suficiente en la base de conocimiento\". "
    "Devuelve EXCLUSIVAMENTE JSON con las claves: "
    "answer (string), sources (string[]), confidence (número 0..1). "
    "No agregues texto fuera del JSON."
)

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no está configurada")
    genai.configure(api_key=api_key)
    model_name = (os.getenv(MODEL_ENV) or DEFAULT_MODEL).strip()
    # 👉 Usar system_instruction en vez de rol 'system'
    return genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)

log = logging.getLogger("gen")

def generate_answer(model, question: str, context: str, sources: list[str]) -> dict:
    # Un solo prompt de usuario; nada de roles 'system'
    user_prompt = f"""
Pregunta: {question}

Contexto:
{context}

Fuentes: {sources}

Responde SOLO con JSON válido.
"""
    resp = model.generate_content(
        user_prompt,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    # Normalización con fallback
    text = resp.text or ""
    try:
        data = json.loads(text)
    except Exception:
        data = {}

    answer = (data.get("answer") or text or "").strip() or "No fue posible generar una respuesta."
    srcs = data.get("sources") or list(sources or [])
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return {"answer": answer, "sources": srcs, "confidence": conf}
