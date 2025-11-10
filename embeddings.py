# app/embeddings.py
"""
Generador de embeddings usando la API de Hugging Face.
Compatible con Render (sin torch ni fastembed).
"""
import os
import httpx
import numpy as np
from app.settings import get_settings

S = get_settings()
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API = "https://api-inference.huggingface.co/pipeline/feature-extraction/" + HF_MODEL
HF_TOKEN = os.getenv("HF_API_TOKEN")

async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not HF_TOKEN:
        raise ValueError("❌ Falta HF_API_TOKEN en el archivo .env")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    out_vectors = []
    async with httpx.AsyncClient(timeout=60) as client:
        for t in texts:
            r = await client.post(HF_API, headers=headers, json={"inputs": t})
            vec = np.mean(np.array(r.json()), axis=0).tolist()
            out_vectors.append(vec)
    return out_vectors
