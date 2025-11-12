# app/rag.py — SOLO nube (HF Inference Providers) + Qdrant + Groq

import os, math, logging, time
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient

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

# Token HF: acepta HF_API_TOKEN o HF_TOKEN
HF_API_TOKEN      = (os.getenv("HF_API_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
HF_EMBED_MODEL    = (os.getenv("HF_EMBED_MODEL") or "intfloat/multilingual-e5-small").strip()

GROQ_API_KEY      = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL        = (os.getenv("GROQ_MODEL") or "gemma2-9b-it").strip()

HF_TIMEOUT = float(os.getenv("HF_TIMEOUT", "60"))
HF_RETRIES = int(os.getenv("HF_RETRIES", "2"))

# Modelos de respaldo por si el principal no está disponible en Providers
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
# Utilidades de vectores
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

def _coerce_to_list_of_vectors(out: Any) -> List[List[float]]:
    """
    Normaliza la salida de feature_extraction a List[List[float]].
    - Si viene [float, ...] → [[...]]
    - Si viene [[float, ...], ...] → tal cual
    - Si viniera tokens x dim (lista de listas) → mean-pooling y normaliza
    """
    if out is None:
        return []
    if isinstance(out, list) and out and isinstance(out[0], (int, float)):
        return [_normalize([float(x) for x in out])]
    if isinstance(out, list) and out and isinstance(out[0], list):
        if out[0] and all(isinstance(x, (int, float)) for x in out[0]):
            return [[float(x) for x in v] for v in out]
        pooled = _mean_pool_2d(out)
        if pooled:
            return [_normalize(pooled)]
    return []

# ─────────────────────────────────────────────────────────────
# HF CLOUD EMBEDDINGS (InferenceClient) con compatibilidad de firma
# ─────────────────────────────────────────────────────────────
# Prefijo para modelos E5/GTE
_E5_LIKE = {
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/e5-base",
    "intfloat/e5-large",
    "alibaba-nlp/gte-base",
    "alibaba-nlp/gte-multilingual-base",
    "gte-small",
    "gte-large",
}
def _maybe_prefix(text: str, model_name: str) -> str:
    name = model_name.lower()
    if any(m in name for m in _E5_LIKE):
        if not text.startswith("passage: "):
            return f"passage: {text}"
    return text

_HF_CLIENT: InferenceClient | None = None
def _hf_client() -> InferenceClient:
    global _HF_CLIENT
    if _HF_CLIENT is None:
        if not HF_API_TOKEN:
            raise RuntimeError("Falta HF_API_TOKEN (o HF_TOKEN) para incrustaciones en la nube.")
        # Cliente general; el modelo se pasa por llamada
        _HF_CLIENT = InferenceClient(token=HF_API_TOKEN)
    return _HF_CLIENT

def hf_embed_texts_cloud(texts: List[str], model: str) -> List[List[float]]:
    """
    Embeddings vía Hugging Face Router (único endpoint oficial desde 2024):
        https://router.huggingface.co/hf-inference
    Usa task='feature-extraction' y hace mean-pooling local.
    Compatible con cualquier modelo público y tokens 'read'.
    """
    if not texts:
        return []
    if not HF_API_TOKEN:
        raise RuntimeError("Falta HF_API_TOKEN en entorno (necesario para router HF).")

    # Prefijo E5/GTE
    tx = [_maybe_prefix(t, model) for t in texts]
    payload = tx[0] if len(tx) == 1 else tx

    url = "https://router.huggingface.co/hf-inference"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
        "X-Wait-For-Model": "true",
        "X-Use-Cache": "true",
    }
    body = {
        "model": model,
        "task": "feature-extraction",
        "inputs": payload,
    }

    try:
        with httpx.Client(timeout=HF_TIMEOUT) as s:
            r = s.post(url, headers=headers, json=body)
            if r.status_code >= 400:
                try:
                    err_body = r.json()
                except Exception:
                    err_body = r.text
                raise RuntimeError(
                    f"Router HF error {r.status_code}: {err_body}"
                )
            data = r.json()
    except Exception as e:
        raise RuntimeError(f"Error al contactar el router HF: {e}")

    # Normalizamos la salida
    vecs = _coerce_to_list_of_vectors(data)
    if not vecs:
        raise RuntimeError("Router devolvió salida vacía o no interpretable.")
    return vecs


def _embed_query_cloud(text: str) -> List[float]:
    try:
        vecs = hf_embed_texts_cloud([text], HF_EMBED_MODEL)
    except Exception as e_cloud:
        log.warning("Falla en router HF, intento local: %s", e_cloud)
        vecs = hf_embed_texts_local([text], HF_EMBED_MODEL)
    return vecs[0]


# ─────────────────────────────────────────────────────────────
# BÚSQUEDA EN QDRANT
# ─────────────────────────────────────────────────────────────
def _search(qvec: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    client = _qdrant()
    hits = client.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=top_k)
    out: List[Dict[str, Any]] = []
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
