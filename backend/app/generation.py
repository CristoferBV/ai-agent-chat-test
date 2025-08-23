import os, json
import google.generativeai as genai

MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-2.0-flash"

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no está configurada")
    genai.configure(api_key=api_key)
    model_name = (os.getenv(MODEL_ENV) or DEFAULT_MODEL).strip()
    return genai.GenerativeModel(model_name)

def generate_answer(model, question: str, context: str, sources: list[str]) -> dict:
    """
    Solicitamos a Gemini que responda EXCLUSIVAMENTE JSON.
    Si el contexto no es suficiente, debe indicarlo explícitamente.
    """
    system = (
        "Eres un asistente que responde únicamente con la información del 'Contexto'. "
        "Si no hay evidencia suficiente, responde: \"No hay información suficiente en la base de conocimiento\". "
        "Devuelve exclusivamente JSON con las claves: answer (string), sources (string[]), confidence (0..1)."
    )
    user = f"Pregunta: {question}\n\nContexto:\n{context}\n\nFuentes: {sources}"

    resp = model.generate_content(
        [
            {"role": "system", "parts": [system]},
            {"role": "user", "parts": [user]},
        ],
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    # Normalización con fallback seguro
    try:
        data = json.loads(resp.text) if resp.text else {}
    except Exception:
        data = {}

    answer = (data.get("answer") or (resp.text or "")).strip() or "No fue posible generar una respuesta."
    srcs = data.get("sources") or list(sources or [])
    try:
        conf = float(data.get("confidence", 0.5))
    except Exception:
        conf = 0.5

    # Clamp confianza a [0,1]
    conf = max(0.0, min(1.0, conf))
    return {"answer": answer, "sources": srcs, "confidence": conf}
