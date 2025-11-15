# app/rag.py — SOLO nube (HF Inference API via router) + Qdrant + Groq
import os, logging, time
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient

# ─────────────────────────────────────────────────────────────
# Carga .env solo si existe y SIN override (no pisar Render)
# ─────────────────────────────────────────────────────────────
_dotenv = find_dotenv(usecwd=True)
if _dotenv:
    load_dotenv(_dotenv, override=False)

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
QDRANT_URL        = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY    = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "ccp_docs").strip()

# Hugging Face Inference (embeddings en la nube)
HF_API_TOKEN      = (os.getenv("HF_API_TOKEN") or "").strip()
HF_EMBED_MODEL    = (os.getenv("HF_EMBED_MODEL")
                     or "intfloat/multilingual-e5-small").strip()
EMBED_BATCH       = int(os.getenv("EMBED_BATCH", "16"))  # batch pequeño

# Groq (LLM Gemma)
GROQ_API_KEY      = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL        = (os.getenv("GROQ_MODEL") or "gemma2-9b-it").strip()

# ─────────────────────────────────────────────────────────────
# LOG
# ─────────────────────────────────────────────────────────────
log = logging.getLogger("rag")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)
log.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────
# Qdrant
# ─────────────────────────────────────────────────────────────
def _qdrant() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Faltan QDRANT_URL o QDRANT_API_KEY en el entorno.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=90)

# ─────────────────────────────────────────────────────────────
# Embeddings por API (Hugging Face Inference via router)
# ─────────────────────────────────────────────────────────────
def _hf_embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Envía un batch de textos a HF Inference API (nuevo router) y devuelve la lista de vectores.
    Usa: https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}
    """
    if not HF_API_TOKEN:
        raise RuntimeError("Falta HF_API_TOKEN en el entorno.")
    if not HF_EMBED_MODEL:
        raise RuntimeError("Falta HF_EMBED_MODEL en el entorno.")

    # 👇 Nuevo endpoint recomendado por HF
    url = f"https://router.huggingface.co/hf-inference/models/{HF_EMBED_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Accept": "application/json",
    }

    payload = {
        "inputs": texts,
        "options": {"wait_for_model": True},
    }

    with httpx.Client(timeout=60) as cli:
        r = cli.post(url, headers=headers, json=payload)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("HF Inference error %s: %s", e.response.status_code, e.response.text)
            raise

        data = r.json()
        vecs: List[List[float]] = []

        # Normalizamos el formato de salida
        if isinstance(data, list) and data and isinstance(data[0], list):
            for item in data:
                if item and isinstance(item[0], list):
                    # Embeddings por token → promedio
                    dim = len(item[0])
                    summed = [0.0] * dim
                    for tok_vec in item:
                        for i, val in enumerate(tok_vec):
                            summed[i] += float(val)
                    vec = [v / float(len(item)) for v in summed]
                    vecs.append(vec)
                else:
                    # Ya es un vector por texto
                    vecs.append([float(v) for v in item])
        else:
            raise RuntimeError(f"Formato inesperado de embeddings HF: {type(data)}")

        return vecs


def _embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        log.info("🧠 Incrustaciones HF (%s) lote %d-%d", HF_EMBED_MODEL, i, i + len(chunk))
        try:
            vecs = _hf_embed_batch(chunk)
            all_vecs.extend(vecs)
        except Exception as e:
            log.exception("Error generando embeddings con HF: %s", e)
            raise
        time.sleep(0.2)
    return all_vecs


def _embed_query(text: str) -> List[float]:
    vecs = _embed_texts([text])
    return vecs[0]

# ─────────────────────────────────────────────────────────────
# Búsqueda en Qdrant
# ─────────────────────────────────────────────────────────────
def _search(qvec: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    client = _qdrant()
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
    out: List[Dict[str, Any]] = []
    for h in hits:
        p = h.payload or {}
        out.append(
            {
                "score": float(h.score),
                "text": p.get("text", ""),
                "page": p.get("page", None),
                "source": p.get("source", ""),
            }
        )
    return out

# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────
SYSTEM = (
    "Eres el asistente oficial de la Cámara de Comercio de Pamplona (Colombia). "
    "Responde SOLO sobre servicios, trámites, horarios y actividades de la Cámara. "
    "Sé claro y específico, evita información inventada. "
    "Si la respuesta no está en las fuentes, dilo de forma directa."
)

def _build_prompt(user_q: str, passages: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for i, p in enumerate(passages, 1):
        snippet = (p["text"] or "").replace("\n", " ").strip()
        if snippet:
            ctx_lines.append(f"[{i}] {snippet}")
    ctx = "\n".join(ctx_lines[:8])

    return (
        f"{SYSTEM}\n\n"
        f"Contexto (fragmentos de la Cámara de Comercio de Pamplona):\n{ctx}\n\n"
        f"Pregunta del usuario: {user_q}\n\n"
        f"Instrucciones para la respuesta:\n"
        f"- Usa SOLO la información del contexto.\n"
        f"- Si hay horarios, direcciones o teléfonos, devuélvelos completos y actualizados.\n"
        f"- Responde en un máximo de 5–7 líneas, formato WhatsApp, usando viñetas cuando ayude.\n"
        f"- Si la información no aparece en el contexto, responde que no cuentas con esos datos.\n"
        f"- No inventes enlaces ni promociones."
    )

# ─────────────────────────────────────────────────────────────
# LLM (Groq)
# ─────────────────────────────────────────────────────────────
def _llm_answer(prompt: str) -> str:
    if not GROQ_API_KEY:
        return "⚠️ Falta GROQ_API_KEY en el entorno."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    with httpx.Client(timeout=60) as cli:
        r = cli.post(url, headers=headers, json=body)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("Groq error %s: %s", e.response.status_code, e.response.text)
            raise
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

# ─────────────────────────────────────────────────────────────
# API principal
# ─────────────────────────────────────────────────────────────
def answer_with_rag(query: str, top_k: int = 5) -> str:
    try:
        if not query or not query.strip():
            return "¿Podrías escribir tu pregunta?"
        log.info("[RAG] Modelo HF (nube): %s | q='%s'", HF_EMBED_MODEL, query[:80])

        qvec = _embed_query(query)
        docs = _search(qvec, top_k=top_k)
        if not docs:
            return "No encontré información sobre eso en la Cámara de Comercio de Pamplona."

        prompt = _build_prompt(query, docs)
        ans = _llm_answer(prompt)
        return ans or "No pude generar respuesta en este momento."
    except Exception as e:
        log.exception("[RAG] Error: %s", e)
        return f"⚠️ Error en RAG: {e}"
