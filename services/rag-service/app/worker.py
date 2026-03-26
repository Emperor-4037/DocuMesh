from celery import Celery
from shared.config import settings
from .ingestion import ingest

celery_app = Celery(
    "rag_tasks",
    broker=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
    backend=f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0",
)
celery_app.conf.task_routes = {"app.worker.ingest_document_task": "rag-queue"}


@celery_app.task(name="app.worker.ingest_document_task", bind=True, max_retries=3)
def ingest_document_task(self, document_content: str, metadata: dict):
    try:
        content_bytes = document_content.encode("utf-8")
        n_chunks = ingest(content_bytes, metadata)
        return {"status": "success", "chunks_indexed": n_chunks, "source": metadata.get("filename")}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
