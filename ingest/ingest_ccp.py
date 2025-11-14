# ingest/ingest_ccp.py
# -*- coding: utf-8 -*-
"""
Ingesta de CAMARA.pdf a Qdrant Cloud usando FastEmbed (local, sin API externa).

Requiere en .env:
QDRANT_URL=https://<tu-id>.<region>.qdrant.io
QDRANT_API_KEY=<tu_api_key>
QDRANT_COLLECTION=ccp_docs
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""

import os
import sys
import uuid
import time
import argparse
from typing import List, Any, Iterable, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

# -------- Cargar .env --------
def root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_env() -> None:
    load_dotenv(os.path.join(root_dir(), ".env"), override=True)

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

# ---------------- Ingesta principal ----------------
def main():
    load_env()

    QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
    COLLECTION = os.getenv("QDRANT_COLLECTION", "ccp_docs").strip()
    EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()

    pdf_default = os.path.join(root_dir(), "knowledge", "CAMARA.pdf")

    parser = argparse.ArgumentParser(description="Ingesta de PDF a Qdrant Cloud (FastEmbed local)")
    parser.add_argument("--pdf", default=pdf_default, help="Ruta del PDF (default: knowledge/CAMARA.pdf)")
    parser.add_argument("--collection", default=COLLECTION, help="Nombre de colección (default: ccp_docs)")
    parser.add_argument("--no-recreate", action="store_true", help="No recrear la colección si ya existe")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote para upsert a Qdrant")
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

    print(f"🧠 Cargando modelo FastEmbed: {EMBED_MODEL}")
    embedder = TextEmbedding(model_name=EMBED_MODEL)
    vectors = [v.tolist() for v in embedder.embed(chunks)]
    if not vectors:
        print("❌ No se generaron embeddings")
        sys.exit(1)
    vector_dim = len(vectors[0])
    print(f"   • Dimensión de vector: {vector_dim}")

    print(f"☁️ Conectando a Qdrant Cloud: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=90)

    if not args.no_recreate:
        print(f"🧺 Recreando colección '{args.collection}'…")
        try:
            client.get_collection(args.collection)
            client.delete_collection(args.collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
        )
    else:
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
