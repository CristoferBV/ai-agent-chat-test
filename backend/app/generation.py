import logging
import os, json
import google.generativeai as genai

MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.0-flash"  # usa este por costo/latencia; cambia si quieres

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
    # En Gemini el 'system' va como system_instruction, no como rol dentro del chat
    return genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)

log = logging.getLogger("gen")

def generate_answer(model, question: str, context: str, sources: list[str]) -> dict:
    # Un solo prompt de usuario; nada de roles 'system'
    user_prompt = f"""
Pregunta: {question}

Contexto:
{context}

Fuentes: {sources}

Responde SOLO con JSON válido con shape:
{{"answer": "texto", "sources": ["url1","url2"], "confidence": 0.8}}
"""

    # Log de diagnóstico
    try:
        log.info("🔎 len(context)=%s chars, sources=%s", len(context or ""), sources)
    except Exception:
        pass

    resp = model.generate_content(
        user_prompt,
        generation_config={
            "temperature": 0.2,
            "candidate_count": 1,
            "response_mime_type": "application/json",
        },
    )

    # Intento de parseo a JSON
    raw = (getattr(resp, "text", "") or "").strip()
    try:
        log.info("📝 raw from Gemini (first 300): %r", raw[:300])
    except Exception:
        pass

    data = {}
    if raw:
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

    # Fallback si no llegó JSON válido con 'answer'
    if not data or "answer" not in data:
        synth = ""
        if context:
            # Respuesta extractiva simple: primeros ~700 chars del contexto
            synth = context.strip().replace("\n\n", "\n")[:700]
        if not synth:
            synth = "No hay información suficiente en la base de conocimiento."
        return {
            "answer": synth,
            "sources": list(sources or []),
            "confidence": 0.4 if synth else 0.1,
        }

    # Normalización si vino JSON válido
    answer = (data.get("answer") or "").strip()
    srcs = data.get("sources") or list(sources or [])
    try:
        conf = float(data.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    conf = max(0.0, min(1.0, conf))
    return {"answer": answer, "sources": srcs, "confidence": conf}
