"""
RAG Ingestion pipeline:
  1. Extract text from PDF/Markdown/TXT
  2. Clean and chunk into overlapping windows
  3. Embed each chunk
  4. Upsert into Qdrant with source metadata
"""
import io
import re
import uuid
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from shared.config import settings

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "documents"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


def extract_text(content: bytes, filename: str) -> str:
    """Supports .txt and .md. Extend with PyMuPDF for PDF."""
    if filename.endswith(".pdf"):
        # Plug in: import fitz; doc = fitz.open(stream=content); return " ".join(p.get_text() for p in doc)
        return "PDF text extraction placeholder"
    return content.decode("utf-8", errors="ignore")


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + size]))
        i += size - overlap
    return chunks


def ensure_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def ingest(content: bytes, metadata: Dict[str, Any], qdrant_client=None) -> int:
    """Full ingestion flow. Returns number of chunks indexed."""
    client = qdrant_client or QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    ensure_collection(client)

    filename = metadata.get("filename", "unknown")
    raw_text = extract_text(content, filename)
    cleaned = clean(raw_text)
    chunks = chunk(cleaned)

    embeddings = EMBED_MODEL.encode(chunks, show_progress_bar=False)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={"text": chk, "source": filename, "chunk_index": idx},
        )
        for idx, (chk, emb) in enumerate(zip(chunks, embeddings))
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)
