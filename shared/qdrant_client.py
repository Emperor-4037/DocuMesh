from qdrant_client import AsyncQdrantClient
from .config import settings

qdrant = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

async def get_qdrant() -> AsyncQdrantClient:
    return qdrant
