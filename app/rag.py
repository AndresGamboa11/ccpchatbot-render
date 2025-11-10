# app/rag.py
import os
import httpx
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from app.settings import get_settings

S = get_settings()

# Cargar modelo de embeddings (una sola vez)
_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def _connect_qdrant() -> QdrantClient:
    return QdrantClient(url=S.URL_QDRANT, api_key=S.CLAVE_API_QDRANT, timeout=60)

def embed_text(texts: List[str]) -> List[List[float]]:
    vecs = _EMBED_MODEL.encode(texts, batch_size=32, show_progress_bar=False)
    return vecs.tolist()

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    client = _connect_qdrant()
    qvec = embed_text([query])[0]
    res = client.search(
        collection_name=S.COLECCION_QDRANT,
        query_vector=qvec,
        limit=k,
        with_payload=True
    )
    out = []
    for p in res:
        payload = p.payload or {}
        out.append({"text": payload.get("text", ""), "page": payload.get("page"), "score": float(p.score)})
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
    user = f"Consulta del usuario: {user_msg}"
    ctx = f"Contexto autorizado (extractos de documentos de la Cámara):\n{ctx_txt if ctx_txt else '- (sin coincidencias relevantes)'}"
    final = (
        f"{system}\n\n"
        f"{ctx}\n\n"
        f"Instrucciones:\n"
        f"- Máximo 6 líneas por respuesta.\n"
        f"- Si no encuentras respuesta en el contexto, dilo de forma clara.\n"
        f"- Incluye números telefónicos o horarios sólo si están en el contexto.\n"
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
        return True, ("¡Hola! 👋 Soy el asistente de la Cámara de Comercio de Pamplona. "
                      "¿En qué puedo ayudarte hoy? Puedo orientarte sobre matrícula, renovación, "
                      "ESAL, conciliación, RUES, certificados y eventos.")
    if any(d in t for d in desped):
        return True, ("¡Con gusto! 😊 Si necesitas algo más de la Cámara de Comercio de Pamplona, "
                      "escríbeme cuando quieras. ¡Que tengas un excelente día!")
    return False, ""

async def call_groq_chat(system_prompt: str) -> str:
    """
    Llama a Groq (Gemma) por HTTP. Devuelve texto plano.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {S.CLAVE_GROQ}",
        "Content-Type": "application/json",
    }
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
            return f"Lo siento, no pude generar respuesta (Groq). Detalle: {data}"

async def answer_with_rag(user_msg: str) -> str:
    # Saludos/despedidas y cortesías
    is_smalltalk, smalltalk = is_greeting_or_farewell(user_msg)
    if is_smalltalk:
        return smalltalk

    # Recuperación y generación
    ctx = retrieve(user_msg, k=5)
    prompt = build_prompt(user_msg, ctx)
    answer = await call_groq_chat(prompt)
    # Si no hay contexto y la IA lo indica ambiguo, reforzar mensaje
    if not ctx:
        answer += "\n\n(No encontré coincidencias en los documentos; si deseas, envíame otra pregunta o comparte el PDF actualizado)."
    return answer
