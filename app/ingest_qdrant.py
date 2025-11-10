# ingest/ingest_ccp.py
# -*- coding: utf-8 -*-
import os, sys, uuid, time, argparse
from typing import List, Any, Iterable, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

def root_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_env() -> None:
    load_dotenv(os.path.join(root_dir(), ".env"))

def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\x00"," ")
    s = "\n".join(l.strip() for l in s.splitlines())
    return " ".join(s.split())

def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> List[str]:
    text = clean_text(text)
    if not text: return []
    if len(text) <= max_chars: return [text]
    chunks, start, step = [], 0, max_chars - overlap
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text): break
        start += step
    return chunks

def chunk_pdf_by_page(reader: PdfReader, max_chars: int = 800, overlap: int = 120):
    results = []
    for i, page in enumerate(reader.pages, start=1):
        txt = clean_text(page.extract_text() or "")
        if not txt: continue
        for ch in chunk_text(txt, max_chars, overlap):
            results.append((ch, i))
    return results

def batched(iterable: Iterable[Any], n: int):
    batch = []
    for it in iterable:
        batch.append(it)
        if len(batch) >= n:
            yield batch; batch=[]
    if batch: yield batch

def main():
    load_env()
    URL_QDRANT = os.getenv("QDRANT_URL","").strip()
    CLAVE_API_QDRANT = os.getenv("QDRANT_API_KEY","").strip()
    COLECCION = os.getenv("QDRANT_COLLECTION","ccp_docs")
    pdf_path_default = os.path.join(root_dir(), "knowledge", "CAMARA.pdf")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=pdf_path_default)
    parser.add_argument("--collection", default=COLECCION)
    parser.add_argument("--no-recreate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not URL_QDRANT or not CLAVE_API_QDRANT:
        print("❌ Faltan QDRANT_URL o QDRANT_API_KEY"); sys.exit(1)
    if not os.path.exists(args.pdf):
        print(f"❌ No existe el PDF: {args.pdf}"); sys.exit(1)

    # ---- Embeddings (FastEmbed) ----
    model_name = "intfloat/multilingual-e5-small"  # 384 dims, multilingüe
    print(f"🧩 Cargando FastEmbed: {model_name}")
    embedder = TextEmbedding(model_name=model_name)
    vector_dim = 384

    print(f"📄 Leyendo PDF: {args.pdf}")
    reader = PdfReader(args.pdf)
    chunks_with_page = chunk_pdf_by_page(reader)
    if not chunks_with_page:
        print("⚠️ Sin texto extraíble."); sys.exit(1)
    chunks = [c for c,_ in chunks_with_page]
    print(f"✅ Fragmentos: {len(chunks)}")

    print("⚙️ Generando embeddings…")
    embs = [v for v in embedder.embed(chunks)]

    print(f"☁️ Conectando Qdrant: {URL_QDRANT}")
    client = QdrantClient(url=URL_QDRANT, api_key=CLAVE_API_QDRANT, timeout=90)

    if not args.no_recreate:
        print(f"🧺 Recreando colección '{args.collection}'…")
        client.recreate_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
        )
    else:
        try:
            client.get_collection(args.collection)
        except Exception:
            client.create_collection(
                collection_name=args.collection,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
            )

    print("⬆️ Subiendo puntos…")
    sent, total = 0, len(chunks)
    t0 = time.time()
    z = list(zip(chunks_with_page, embs))
    for batch in batched(z, args.batch_size):
        ids = [str(uuid.uuid4()) for _ in batch]
        vectors = [v for (_, v) in batch]
        payloads = [{"text": ch, "page": pg, "source": os.path.basename(args.pdf)} for ((ch, pg), _) in batch]
        client.upsert(args.collection, points=models.Batch(ids=ids, vectors=vectors, payloads=payloads))
        sent += len(ids)
        print(f"   • {sent}/{total}")

    count = client.count(args.collection, exact=True).count
    print(f"✅ Ingesta OK. Colección: {args.collection} | Puntos: {count} | ⏱ {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
