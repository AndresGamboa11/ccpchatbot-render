# app/embeddings.py
"""
Generador de embeddings usando la API de Hugging Face.
Compatible con Render (sin torch ni fastembed).
"""
# app/embeddings.py
# app/embeddings.py
import os
import httpx
import numpy as np
from typing import List
from app.settings import get_settings
from app.hf_embed import hf_embed

# ...
vectors = hf_embed(textos, model=HF_EMBED_MODEL, batch_size=32)

S = get_settings()

HF_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"
HF_API = f"https://api-inference.huggingface.co/models/{HF_MODEL}"or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"
HF_TOKEN = os.getenv("HF_API_TOKEN", "").strip()

def _pool_to_sentence(vec_json):
    """
    El API puede devolver:
    - [dim]                              -> ya es un embedding de frase
    - [[dim], [dim], ...]                -> token embeddings → hacemos mean pooling
    - [[[...]], [[...]], ...] (batch)    -> lo maneja la función principal
    """
    if not isinstance(vec_json, list):
        raise ValueError("Respuesta inesperada de HF API")

    # Caso ya-pooleado: lista de floats
    if vec_json and isinstance(vec_json[0], (int, float)):
        return vec_json

    # Caso token-level: lista de listas de floats
    arr = np.array(vec_json, dtype=float)
    # mean sobre el eje de tokens → vector [dim]
    return arr.mean(axis=0).tolist()

async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Genera embeddings vía Hugging Face Inference API (nuevo endpoint /models).
    Sin torch/fastembed. Compatible con Render.
    """
    if not HF_TOKEN:
        raise ValueError("❌ Falta HF_API_TOKEN en el .env")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    vectors: List[List[float]] = []

    async with httpx.AsyncClient(timeout=60) as client:
        for t in texts:
            payload = {
                "inputs": t,
                "options": {"wait_for_model": True},   # arranca el modelo si está dormido
            }
            r = await client.post(HF_API, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            # Si el modelo devuelve batch anidado, desempacamos
            if isinstance(data, list) and data and isinstance(data[0], list) and data and isinstance(data[0][0], list):
                # ej. [[[...token dims...]]]  -> tomar el primero del batch y hacer mean pooling
                vec = _pool_to_sentence(data[0])
            else:
                vec = _pool_to_sentence(data)
            vectors.append(vec)

    return vectors
