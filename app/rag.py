# app/rag.py
import httpx
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient

from app.embeddings import embed_texts  # <- usamos la API de HF (async)
from app.settings import get_settings

S = get_settings()


def _connect_qdrant() -> QdrantClient:
    return QdrantClient(url=S.URL_QDRANT, api_key=S.CLAVE_API_QDRANT, timeout=60)


async def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Recupera los k fragmentos más relevantes desde Qdrant.
    Genera el embedding del query usando Hugging Face (embed_texts) y ejecuta la búsqueda.
    """
    client = _connect_qdrant()
    qvec = (await embed_texts([query]))[0]  # <- embedding async
    res = client.search(
        collection_name=S.COLECCION_QDRANT,
        query_vector=qvec,
        limit=k,
        with_payload=True,
    )
    out: List[Dict[str, Any]] = []
    for p in res:
        payload = p.payload or {}
        out.append(
            {
                "text": payload.get("text", ""),
                "page": payload.get("page"),
                "score": float(p.score),
            }
        )
    return out


def build_prompt(user_msg: str, ctx_snippets: List[Dict[str, Any]]) -> str:
    ctx_txt = "\n\n".join([f"- (pág.{c.get('page','?')}) {c['text']}" for c in ctx_snippets])
    system = (
        "Eres el asistente virtual OFICIAL de la Cámara de Comercio de Pamplona (Colombia). "
        "Responde con información precisa y breve, en español, usando SÓLO el contexto provisto. "
        "Si el usuario pregunta algo general (saludos, despedidas, cortesías), responde de forma cordial y breve. "
        "Si el usuario pide información fuera del ámbito de la Cámara, responde: "
        "\"Lo siento, solo puedo ayudarte con información de la Cámara de Comercio de Pamplona.\" "
        "Cuando el tema sea de la Cámara, usa listas cortas y evita inventar datos."
    )
    final = (
        f"{system}\n\n"
        f"Contexto autorizado (extractos):\n{ctx_txt if ctx_txt else '- (sin coincidencias relevantes)'}\n\n"
        f"Instrucciones:\n"
        f"- Máximo 6 líneas por respuesta.\n"
        f"- Si no encuentras respuesta en el contexto, dilo de forma clara.\n"
        f"- Incluye teléfonos/horarios solo si están en el contexto.\n"
        f"- No repitas el contexto.\n"
        f"Usuario: {user_msg}\n"
        f"Respuesta:"
    )
    return final


def is_greeting_or_farewell(text: str) -> Tuple[bool, str]:
    t = (text or "").lower()
    saludos = ["hola", "buenos días", "buenas tardes", "buenas noches", "qué tal", "buen día"]
    desped = ["gracias", "muchas gracias", "hasta luego", "chao", "adiós", "nos vemos"]
    if any(s in t for s in saludos):
        return True, (
            "¡Hola! 👋 Soy el asistente de la Cámara de Comercio de Pamplona. "
            "¿En qué puedo ayudarte? Puedo orientarte sobre matrícula, renovación, "
            "ESAL, conciliación, RUES, certificados y eventos."
        )
    if any(d in t for d in desped):
        return True, (
            "¡Con gusto! 😊 Si necesitas algo más de la Cámara de Comercio de Pamplona, "
            "escríbeme cuando quieras. ¡Que tengas un excelente día!"
        )
    return False, ""


async def call_groq_chat(system_prompt: str) -> str:
    """
    Llama a Groq (Gemma) por HTTP. Devuelve texto plano.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {S.CLAVE_GROQ}", "Content-Type": "application/json"}
    payload = {
        "model": S.MODELO_GROQ,
        "messages": [
            {"role": "system", "content": "Sigue las instrucciones y responde solo en español."},
            {"role": "user", "content": system_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 350,
    }
    async with httpx.AsyncClient(timeout=S.TIMEOUT_HTTP) as client:
        r = await client.post(url, headers=headers, json=payload)
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return f"No pude generar respuesta (Groq). Detalle: {data}"


async def answer_with_rag(user_msg: str) -> str:
    # Small talk (saludos/despedidas)
    ok_small, smalltalk = is_greeting_or_farewell(user_msg)
    if ok_small:
        return smalltalk

    # Recuperación + generación
    ctx = await retrieve(user_msg, k=5)  # <- ahora es async
    prompt = build_prompt(user_msg, ctx)
    answer = await call_groq_chat(prompt)

    if not ctx:
        answer += "\n\n(No encontré coincidencias en los documentos; intenta con otra consulta o actualiza el PDF)."
    return answer
