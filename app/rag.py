# app/rag.py
import os, math, logging
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)

# ---- ENV ----
QDRANT_URL       = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY   = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION= (os.getenv("QDRANT_COLLECTION") or "ccp_docs").strip()
HF_EMBED_MODEL   = (os.getenv("HF_EMBED_MODEL") or "intfloat/multilingual-e5-small").strip()
GROQ_API_KEY     = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL       = (os.getenv("GROQ_MODEL") or "gemma2-9b-it").strip()

# ---- LOGS ----
logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)

# ---- Qdrant client ----
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=90)

# ---- Embeddings locales (dim 384 con e5-small) ----
try:
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer(HF_EMBED_MODEL, device="cpu")
    _EMBED_DIM = _st.get_sentence_embedding_dimension()
except Exception as e:
    raise RuntimeError(f"❌ No pude cargar SentenceTransformer({HF_EMBED_MODEL}): {e}")

def _embed_query(text: str) -> List[float]:
    vec = _st.encode([text], normalize_embeddings=True, batch_size=1, convert_to_numpy=False)[0]
    return [float(x) for x in vec]

# ---- Buscador en Qdrant ----
def _search(qvec: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    hits = _client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
    out = []
    for h in hits:
        out.append({
            "score": float(h.score),
            "text": (h.payload or {}).get("text", ""),
            "page": (h.payload or {}).get("page", None),
            "source": (h.payload or {}).get("source", ""),
        })
    return out

# ---- Construcción de prompt (breve y exacto para WhatsApp) ----
SYSTEM = (
    "Eres el asistente oficial de la Cámara de Comercio de Pamplona (Colombia). "
    "Responde SOLO sobre servicios, trámites y horarios de la Cámara. "
    "Sé conciso (WhatsApp), con viñetas cuando ayude. Si no está en las fuentes, di con claridad que no tienes esa información."
)

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


# ---- Llamada a Groq (vía endpoint OpenAI-compatible) ----
import httpx

def _llm_answer(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Falta GROQ_API_KEY en el entorno. No puedo generar respuesta."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    with httpx.Client(timeout=60) as cli:
        r = cli.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# ---- API principal: answer_with_rag ----
def answer_with_rag(query: str, top_k: int = 5) -> str:
    try:
        if not query or not query.strip():
            return "¿Podrías escribir tu pregunta?"
        logger.info(f"[RAG] Modelo embeddings: {HF_EMBED_MODEL} (dim={_EMBED_DIM}) | Coll={QDRANT_COLLECTION}")
        qvec = _embed_query(query)
        docs = _search(qvec, top_k=top_k)
        if not docs:
            return "No encontré información sobre eso en la Cámara de Comercio de Pamplona."
        prompt = _build_prompt(query, docs)
        ans = _llm_answer(prompt)
        if not ans:
            return "No pude generar respuesta en este momento."
        return ans
    except Exception as e:
        logger.exception(f"[RAG] Error: {e}")
        return f"⚠️ Error en RAG: {e}"
