# app/rag.py — SOLO nube (HF Inference API) + Qdrant + Groq
import os, math, logging, time
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

HF_API_TOKEN      = (os.getenv("HF_API_TOKEN") or "").strip()
HF_EMBED_MODEL    = (os.getenv("HF_EMBED_MODEL") or "intfloat/multilingual-e5-small").strip()

GROQ_API_KEY      = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL        = (os.getenv("GROQ_MODEL") or "gemma2-9b-it").strip()

HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))
HF_RETRIES = int(os.getenv("HF_RETRIES", "2"))

# Modelos de respaldo por si el principal no tiene Inference API activo
HF_EMBED_FALLBACKS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "Alibaba-NLP/gte-multilingual-base",
    "intfloat/multilingual-e5-base",
]

# ─────────────────────────────────────────────────────────────
# LOG
# ─────────────────────────────────────────────────────────────
log = logging.getLogger("rag")
if not log.handlers:
    logging.basicConfig(level=logging.INFO)
log.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────
# Qdrant (lazy) — para fallar con mensaje claro si faltan env
# ─────────────────────────────────────────────────────────────
def _qdrant() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("Faltan QDRANT_URL o QDRANT_API_KEY en el entorno.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=90)

# ─────────────────────────────────────────────────────────────
# HF CLOUD EMBEDDINGS (solo nube)
# ─────────────────────────────────────────────────────────────
def _mean_pool_2d(vectors_2d: List[List[float]]) -> List[float]:
    if not vectors_2d:
        return []
    dims = len(vectors_2d[0])
    sums = [0.0] * dims
    count = 0
    for tok in vectors_2d:
        if len(tok) != dims:
            continue
        for i, val in enumerate(tok):
            sums[i] += float(val)
        count += 1
    if count == 0:
        return []
    return [s / count for s in sums]

def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum((x * x) for x in vec)) or 1.0
    return [x / norm for x in vec]

def _parse_hf_feature_response(data: Any) -> List[List[float]]:
    # Soporta: tokens x dim → mean-pooling; vector 1D; {"data":[{"embedding":[...]}]}
    if isinstance(data, list) and not data:
        return []
    if isinstance(data, list):
        if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
            return [_normalize(_mean_pool_2d(data))]
        if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
            return [_normalize(_mean_pool_2d(data[0]))]
        if data and isinstance(data[0], (int, float)):
            return [_normalize([float(x) for x in data])]
    if isinstance(data, dict) and "data" in data:
        out = []
        for item in data["data"]:
            if isinstance(item, dict) and "embedding" in item:
                out.append(_normalize([float(x) for x in item["embedding"]]))
        if out:
            return out
    return []

def _hf_post(url: str, token: str, payload: dict, timeout: float):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as cli:
        r = cli.post(url, headers=headers, json=payload)
    return r

def _try_model_embeddings(texts: List[str], model: str, token: str) -> List[List[float]]:
    """
    Intenta primero el endpoint /embeddings/{model} y luego
    /pipeline/feature-extraction/{model}. Si ambos fallan, levanta excepción.
    """
    # 1) Nuevo endpoint de embeddings
    url1 = f"https://api-inference.huggingface.co/embeddings/{model}"
    r1 = _hf_post(url1, token, {"inputs": texts, "options": {"wait_for_model": True}}, HF_TIMEOUT)
    if r1.is_success:
        data = r1.json()
        vecs = _parse_hf_feature_response(data)
        if vecs:
            return vecs
    else:
        log.warning("httpx: Solicitud HTTP: POST %s %s", url1, r1.reason_phrase or r1.status_code)

    # 2) Fallback al pipeline feature-extraction
    url2 = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
    r2 = _hf_post(url2, token, {"inputs": texts, "options": {"wait_for_model": True}}, HF_TIMEOUT)
    if r2.is_success:
        data = r2.json()
        vecs = _parse_hf_feature_response(data)
        if vecs:
            return vecs
    else:
        log.warning("httpx: Solicitud HTTP: POST %s %s", url2, r2.reason_phrase or r2.status_code)

    # Ninguno devolvió embeddings: provoca raise_for_status del último response
    r = r2 if not r1.is_success else r1
    r.raise_for_status()

def hf_embed_texts_cloud(texts: List[str], model: str, token: str) -> List[List[float]]:
    """
    Intenta con HF_EMBED_MODEL; si falla (410/4xx/5xx), rota por HF_EMBED_FALLBACKS.
    """
    if not texts:
        return []

    models_to_try = [model] + [m for m in HF_EMBED_FALLBACKS if m != model]
    last_err = None

    for m in models_to_try:
        for attempt in range(1, HF_RETRIES + 2):
            try:
                vecs = _try_model_embeddings(texts, m, token)
                if vecs:
                    if m != model:
                        log.info("[HF] usando modelo de respaldo: %s", m)
                    return vecs
            except Exception as e:
                last_err = e
                log.warning("trap: [HF] intento %s con %s falló: %s", attempt, m, e)
                time.sleep(1.2 * attempt)
                continue

    raise RuntimeError(f"HF embeddings falló: {last_err}")

def _embed_query_cloud(text: str) -> List[float]:
    if not HF_API_TOKEN:
        raise RuntimeError("Falta HF_API_TOKEN para incrustaciones en la nube.")
    vecs = hf_embed_texts_cloud([text], HF_EMBED_MODEL, HF_API_TOKEN)
    return vecs[0]

# ─────────────────────────────────────────────────────────────
# BÚSQUEDA EN QDRANT
# ─────────────────────────────────────────────────────────────
def _search(qvec: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    client = _qdrant()
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
    out = []
    for h in hits:
        p = h.payload or {}
        out.append({
            "score": float(h.score),
            "text": p.get("text", ""),
            "page": p.get("page", None),
            "source": p.get("source", ""),
        })
    return out

# ─────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────
SYSTEM = (
    "Eres el asistente oficial de la Cámara de Comercio de Pamplona (Colombia). "
    "Responde SOLO sobre servicios, trámites, horarios y actividades de la Cámara. "
    "Sé breve (WhatsApp), con viñetas. Si no está en las fuentes, dilo claramente."
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
        f"Contexto:\n{ctx}\n\n"
        f"Pregunta del usuario: {user_q}\n\n"
        f"Instrucciones:\n"
        f"- Usa SOLO el contexto.\n"
        f"- Si hay horarios, devuélvelos completos.\n"
        f"- Formato conciso, con viñetas cuando ayude.\n"
        f"- No inventes datos ni enlaces."
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
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

# ─────────────────────────────────────────────────────────────
# API PRINCIPAL
# ─────────────────────────────────────────────────────────────
def answer_with_rag(query: str, top_k: int = 5) -> str:
    """
    Función SÍNCRONA (no usar await).
    Hace: embed cloud → search Qdrant → LLM Groq.
    """
    try:
        if not query or not query.strip():
            return "¿Podrías escribir tu pregunta?"
        log.info("[RAG] HF model (nube): %s | q='%s'", HF_EMBED_MODEL, query[:80])

        qvec = _embed_query_cloud(query)
        docs = _search(qvec, top_k=top_k)
        if not docs:
            return "No encontré información sobre eso en la Cámara de Comercio de Pamplona."

        prompt = _build_prompt(query, docs)
        ans = _llm_answer(prompt)
        return ans or "No pude generar respuesta en este momento."
    except Exception as e:
        log.exception("[RAG] Error: %s", e)
        return f"⚠️ Error en RAG: {e}"
