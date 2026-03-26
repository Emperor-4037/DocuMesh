"""
Full RAG Query Pipeline:
  1. Embed the user query with sentence-transformers
  2. Retrieve top-k semantic matches from Qdrant
  3. Rerank with a cross-encoder for precision
  4. Build a grounded prompt from top passages
  5. Generate an answer (currently mocked; swap in LLM call)
"""
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from shared.config import settings

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
RERANK_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION = "documents"


def embed(text: str) -> List[float]:
    return EMBED_MODEL.encode(text).tolist()


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
        {"text": h.payload.get("text", ""), "source": h.payload.get("source", ""), "score": h.score}
        for h in hits
    ]


def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """Cross-encoder reranking to improve precision."""
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = RERANK_MODEL.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [item for _, item in ranked]


def build_prompt(query: str, passages: List[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {p['source']}]\n{p['text']}" for p in passages[:5]
    )
    return (
        f"You are a helpful assistant. Answer the question using ONLY the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


def generate_answer(prompt: str) -> str:
    """
    MVP: returns the prompt as a mock answer.
    Production: swap for an LLM call (OpenAI, Anthropic, local Ollama, etc.)
    """
    return f"[LLM Answer placeholder]\n\nPrompt sent to model:\n{prompt[:500]}..."


def rag_query_pipeline(query: str, top_k: int = 5, qdrant_client=None) -> Tuple[str, List[dict]]:
    candidates = retrieve(query, top_k=max(top_k * 2, 10), qdrant_client=qdrant_client)
    reranked = rerank(query, candidates)
    top_passages = reranked[:top_k]
    prompt = build_prompt(query, top_passages)
    answer = generate_answer(prompt)
    return answer, top_passages
