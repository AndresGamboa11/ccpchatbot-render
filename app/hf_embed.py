# app/hf_embed.py
import os, math
from typing import List
from huggingface_hub import InferenceClient

HF_API_TOKEN = (os.getenv("HF_API_TOKEN") or "").strip()
HF_EMBED_MODEL = (os.getenv("HF_EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2").strip()

# Modelos tipo E5/GTE requieren prefijo "query: " o "passage: ". Para RAG, usa "passage: "
_E5_LIKE = {"intfloat/multilingual-e5-base", "intfloat/e5-base", "intfloat/e5-large",
            "Alibaba-NLP/gte-base", "Alibaba-NLP/gte-multilingual-base", "gte-small", "gte-large"}

def _maybe_prefix(text: str, model_name: str) -> str:
    name = model_name.lower()
    if any(m.lower() in name for m in _E5_LIKE):
        # Para documentos (corpus) usamos "passage: "
        if not text.startswith("passage: "):
            return f"passage: {text}"
    return text

def hf_embed(texts: List[str], model: str = HF_EMBED_MODEL, batch_size: int = 32) -> List[List[float]]:
    """
    Devuelve una lista de vectores (uno por texto).
    Compatible con cambios de firma de feature_extraction (text vs posicional).
    """
    if not texts:
        return []
    client = InferenceClient(token=HF_API_TOKEN)

    out: List[List[float]] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        chunk = texts[i:i+batch_size]
        # Prefijo para E5/GTE
        chunk = [_maybe_prefix(t, model) for t in chunk]

        # Llamada tolerante a versión
        try:
            # Firmas nuevas (text=..., pooling, normalize)
            vecs = client.feature_extraction(
                model=model,
                text=chunk,
                pooling="mean",
                normalize=True,
                truncate=True,  # evita secuencias largas
            )
        except TypeError:
            # Firmas antiguas (primer arg posicional, sin keywords)
            vecs = client.feature_extraction(
                chunk,
                model=model,
                pooling="mean",
                normalize=True,
                truncate=True,
            )

        # HF puede devolver un solo vector cuando len(chunk)==1
        if isinstance(vecs[0], (int, float)):
            out.append(vecs)  # type: ignore[arg-type]
        else:
            out.extend(vecs)  # type: ignore[arg-type]
    return out
