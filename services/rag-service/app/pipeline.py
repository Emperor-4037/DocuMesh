"""
Full RAG Query Pipeline:
  1. Embed the user query with sentence-transformers (all-MiniLM-L6-v2)
  2. Retrieve top-k semantic matches from Qdrant
  3. Rerank with a cross-encoder (ms-marco-MiniLM-L-6-v2)
  4. Build grounded context block from top passages
  5. Generate answer via TinyLlama-1.1B-Chat (4-bit quantized on GPU)
"""
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from shared.config import settings
from shared.model_utils import DEVICE
from .llm import generate_answer

# Embed + rerank models are fast and lightweight — load at module level
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)
COLLECTION = "documents"


def embed(text: str) -> List[float]:
    return EMBED_MODEL.encode(text, convert_to_tensor=False).tolist()


def retrieve(query: str, top_k: int = 10, qdrant_client=None) -> List[dict]:
    """Semantic retrieval from Qdrant."""
    client = qdrant_client or QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    q_vec = embed(query)
    hits = client.search(
        collection_name=COLLECTION,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": h.payload.get("text", ""),
            "source": h.payload.get("source", "unknown"),
            "chunk_index": h.payload.get("chunk_index", 0),
            "score": h.score,
        }
        for h in hits
    ]


def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """Cross-encoder reranking for precision over cosine similarity."""
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = RERANK_MODEL.predict(pairs, convert_to_numpy=True)
    ranked = sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked]


def rag_query_pipeline(
    query: str, top_k: int = 5, qdrant_client=None
) -> Tuple[str, List[dict]]:
    """
    End-to-end RAG: retrieve → rerank → generate.
    Returns (answer_text, top_passages_with_metadata).
    """
    # Over-retrieve for reranker to select from
    candidates = retrieve(query, top_k=max(top_k * 3, 15), qdrant_client=qdrant_client)
    reranked = rerank(query, candidates)
    top_passages = reranked[:top_k]

    # Real LLM generation (TinyLlama, 4-bit quantized)
    answer = generate_answer(query, top_passages)

    return answer, top_passages
