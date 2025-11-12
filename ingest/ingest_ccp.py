# ingest/ingest_ccp.py
# -*- coding: utf-8 -*-
"""
Ingesta de CAMARA.pdf a Qdrant Cloud usando embeddings locales (sentence-transformers)
con fallback a Hugging Face Inference API (endpoint nuevo).

Requiere en .env:
QDRANT_URL=https://<tu-id>.<region>.qdrant.io
QDRANT_API_KEY=<tu_api_key>
QDRANT_COLLECTION=ccp_docs
HF_API_TOKEN=hf_xxx (https://huggingface.co/settings/tokens)
HF_EMBED_MODEL=intfloat/multilingual-e5-small   (recomendado, multilingüe)
"""

import os
import sys
import uuid
import time
import argparse
from typing import List, Any, Iterable, Tuple

import httpx
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from app.hf_embed import hf_embed

# ...
vectors = hf_embed(textos, model=HF_EMBED_MODEL, batch_size=32)
# -------- Cargar .env (forzar override) --------
load_dotenv(override=True)

HF_API_TOKEN   = os.getenv("HF_API_TOKEN", "").strip()
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-small").strip()

print("📦 Modelo embeddings activo:", HF_EMBED_MODEL)

# ---------------- Utilidades ----------------
def root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_env() -> None:
    load_dotenv(os.path.join(root_dir(), ".env"))

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = "\n".join(line.strip() for line in s.splitlines())
    return " ".join(s.split())

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    step = max_chars - overlap if max_chars > overlap else max_chars
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks

def chunk_pdf_by_page(reader: PdfReader, max_chars: int = 800, overlap: int = 120) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")
        if not page_text:
            continue
        for ch in chunk_text(page_text, max_chars=max_chars, overlap=overlap):
            results.append((ch, i))
    return results

def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

# --------------- Embeddings (local + fallback HF) ---------------
def embed_texts_sync(texts, hf_token: str, model: str):
    """
    Estrategia:
      A) Local con sentence-transformers (rápido/estable; sin API)
      B) Fallback a Hugging Face /pipeline/feature-extraction/{model}
    """
    if not texts:
        return []

    # ---------- A) Local: sentence-transformers ----------
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model, device="cpu")
        vecs = st.encode(texts, normalize_embeddings=True, batch_size=32, convert_to_numpy=False)
        return [[float(x) for x in v] for v in vecs]
    except Exception as e_local:
        print("⚠️ Embeddings locales fallaron, intento con HF:", str(e_local))

    # ---------- B) HF nuevo endpoint ----------
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

    r = httpx.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # parseo simple (mean-pooling si vienen por tokens)
    def mean_pool(mat):
        if not mat:
            return []
        dims = len(mat[0])
        sums = [0.0] * dims
        n = 0
        for row in mat:
            if len(row) != dims:
                continue
            for i, v in enumerate(row):
                sums[i] += float(v)
            n += 1
        return [s / (n or 1) for s in sums]

    if isinstance(data, list):
        # [[tok_dim], [tok_dim], ...] => tokens de una entrada
        if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], (int, float)):
            return [mean_pool(data)]
        # [ [ [tok_dim], ... ] ] => lista envolvente con tokens
        if data and isinstance(data[0], list) and data[0] and isinstance(data[0][0], list):
            return [mean_pool(data[0])]
        # [float, float, ...] => vector directo
        if data and isinstance(data[0], (int, float)):
            return [[float(x) for x in data]]

    if isinstance(data, dict) and "data" in data:
        out = []
        for item in data["data"]:
            if isinstance(item, dict) and "embedding" in item:
                out.append([float(x) for x in item["embedding"]])
        if out:
            return out

    raise RuntimeError("Formato de respuesta de HF no reconocido.")

# ---------------- Ingesta principal ----------------
def main():
    load_env()

    QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
    COLLECTION = os.getenv("QDRANT_COLLECTION", "ccp_docs").strip()
    HF_TOKEN = os.getenv("HF_API_TOKEN", "").strip()

    pdf_default = os.path.join(root_dir(), "knowledge", "CAMARA.pdf")

    parser = argparse.ArgumentParser(description="Ingesta de PDF a Qdrant Cloud (HF embeddings)")
    parser.add_argument("--pdf", default=pdf_default, help="Ruta del PDF (default: knowledge/CAMARA.pdf)")
    parser.add_argument("--collection", default=COLLECTION, help="Nombre de colección (default: ccp_docs)")
    parser.add_argument("--no-recreate", action="store_true", help="No recrear la colección si ya existe")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote para upsert")
    args = parser.parse_args()

    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ Faltan QDRANT_URL o QDRANT_API_KEY en .env")
        sys.exit(1)
    if not os.path.exists(args.pdf):
        print(f"❌ No se encontró el PDF: {args.pdf}")
        sys.exit(1)

    print(f"📄 Leyendo PDF: {args.pdf}")
    reader = PdfReader(args.pdf)
    chunks_with_page = chunk_pdf_by_page(reader, max_chars=800, overlap=120)
    if not chunks_with_page:
        print("⚠️ El PDF no tiene texto extraíble (¿escaneado sin OCR?).")
        sys.exit(1)
    chunks = [c for c, _ in chunks_with_page]
    print(f"✅ Fragmentos: {len(chunks)}")

    print(f"🧠 Generando embeddings con HF: {HF_EMBED_MODEL}")
    vectors = embed_texts_sync(chunks, HF_TOKEN, HF_EMBED_MODEL)
    if not vectors:
        print("❌ No se generaron embeddings")
        sys.exit(1)
    vector_dim = len(vectors[0])
    print(f"   • Dimensión de vector: {vector_dim}")

    print(f"☁️ Conectando a Qdrant Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=90)

    if not args.no_recreate:
        print(f"🧺 Recreando colección '{args.collection}'…")
        # recreación segura (sin usar recreate_collection, que está deprecado)
        try:
            client.get_collection(args.collection)
            client.delete_collection(args.collection)
        except Exception:
            # si no existe, ignorar
            pass
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
        )
    else:
        # usar colección existente o crear si no existe
        try:
            client.get_collection(args.collection)
            print(f"📦 Usando colección existente: {args.collection}")
        except Exception:
            print(f"📦 Colección no existe. Creando '{args.collection}'…")
            client.create_collection(
                collection_name=args.collection,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
            )

    print("⬆️ Subiendo puntos…")
    sent, total = 0, len(chunks)
    t0 = time.time()
    for batch in batched(list(zip(chunks_with_page, vectors)), args.batch_size):
        ids = [str(uuid.uuid4()) for _ in batch]
        vecs = [v for (_, v) in batch]
        payloads = [{"text": ch, "page": pg, "source": os.path.basename(args.pdf)} for ((ch, pg), _) in batch]
        client.upsert(collection_name=args.collection, points=models.Batch(ids=ids, vectors=vecs, payloads=payloads))
        sent += len(ids)
        print(f"   • {sent}/{total}")

    count = client.count(args.collection, exact=True).count
    print(f"✅ Ingesta completada. Colección: {args.collection} | Puntos: {count} | ⏱ {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
