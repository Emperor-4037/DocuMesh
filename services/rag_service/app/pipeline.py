"""
Full RAG Query Pipeline:
  1. Embed query with sentence-transformers
  2. Retrieve from Qdrant (semantic + optional hybrid)
  3. Rerank with cross-encoder
  4. Generate grounded answer via LLM
  5. Parse citations from LLM output
"""
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from shared.config import settings
from shared.model_config import RAG_EMBED_CONFIG, RAG_RERANK_CONFIG
from shared.model_utils import DEVICE
from shared.logging import setup_logging
from .llm import generate_answer

logger = setup_logging("rag-pipeline")

# Load embedding + reranking models at module level (lightweight, fast)
EMBED_MODEL = SentenceTransformer(RAG_EMBED_CONFIG.base_model, device=DEVICE)
RERANK_MODEL = CrossEncoder(RAG_RERANK_CONFIG.base_model, device=DEVICE)
COLLECTION = "documents"


def embed(text: str) -> List[float]:
    return EMBED_MODEL.encode(text, convert_to_tensor=False).tolist()


def retrieve(query: str, top_k: int = 10, qdrant_client=None) -> List[dict]:
    """Semantic retrieval from Qdrant."""
    client = qdrant_client or QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    q_vec = embed(query)

    try:
        hits = client.search(
            collection_name=COLLECTION,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        logger.error(f"Qdrant retrieval failed: {e}")
        return []

    return [
        {
            "text": h.payload.get("text", ""),
            "source": h.payload.get("source", "unknown"),
            "chunk_index": h.payload.get("chunk_index", 0),
            "page": h.payload.get("page", None),
            "score": h.score,
        }
        for h in hits
    ]


def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """Cross-encoder reranking for precision."""
    if not candidates:
        return []

    try:
        pairs = [(query, c["text"]) for c in candidates]
        scores = RERANK_MODEL.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(scores.tolist(), candidates), key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked]
    except Exception as e:
        logger.warning(f"Reranking failed, falling back to retrieval order: {e}")
        return candidates


def rag_query_pipeline(
    query: str, top_k: int = 5, qdrant_client=None
) -> Tuple[str, List[dict]]:
    """
    End-to-end RAG: retrieve → rerank → generate.
    Gracefully degrades if reranker fails.
    """
    # Over-retrieve for reranker to select from
    candidates = retrieve(query, top_k=max(top_k * 3, 15), qdrant_client=qdrant_client)

    if not candidates:
        return "I cannot answer this question as no relevant documents were found.", []

    reranked = rerank(query, candidates)
    top_passages = reranked[:top_k]

    # Generate answer via LLM
    answer = generate_answer(query, top_passages)

    return answer, top_passages
