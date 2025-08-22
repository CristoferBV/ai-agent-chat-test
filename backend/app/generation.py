import os, json
import google.generativeai as genai

MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-1.5-flash"

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no está configurada")
    genai.configure(api_key=api_key)
    model_name = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    return genai.GenerativeModel(model_name)

def generate_answer(model, question: str, context: str, sources: list[str]) -> dict:
    """
    Pedimos a Gemini que responda EN JSON ESTRICTO.
    Si no hay contexto suficiente, debe indicarlo.
    """
    system = (
        "Eres un asistente empresarial. Responde SOLO con la información provista en 'Contexto'. "
        "Si el contexto no es suficiente, di explícitamente que no hay información suficiente. "
        "Devuelve EXCLUSIVAMENTE JSON con las claves: "
        "answer (string), sources (string[]), confidence (número entre 0 y 1)."
    )
    prompt_user = (
        f"Pregunta: {question}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Fuentes: {sources}"
    )

    resp = model.generate_content(
        [
            {"role": "system", "parts": [system]},
            {"role": "user", "parts": [prompt_user]},
        ],
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    try:
        # Con response_mime_type, resp.text debería ser JSON estricto.
        return json.loads(resp.text)
    except Exception:
        # Fallback: envolvemos en formato esperado si el modelo no cumplió.
        text = (resp.text or "").strip()
        return {
            "answer": text if text else "No fue posible generar una respuesta.",
            "sources": list(sources or []),
            "confidence": 0.5,
        }
