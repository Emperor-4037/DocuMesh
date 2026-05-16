import asyncio
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from shared.schemas import RAGQueryRequest, RAGQueryResponse, ChunkMetadata
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .pipeline import rag_query_pipeline
from .llm import is_ready as llm_ready
from .worker import ingest_document_task

logger = setup_logging("rag-service")
app = FastAPI(title="RAG Service")
instrument_app(app, "rag-service")


@app.post("/query", response_model=RAGQueryResponse)
async def process_query(request: RAGQueryRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing RAG query", extra={"trace_id": trace_id, "query": request.query[:80]})

    loop = asyncio.get_event_loop()
    with LatencyTracker("rag-service", "/query"):
        answer, passages = await loop.run_in_executor(
            None, rag_query_pipeline, request.query, request.top_k
        )

    sources = [
        ChunkMetadata(source=p["source"], page=p.get("chunk_index"))
        for p in passages
    ]
    return RAGQueryResponse(answer=answer, sources=sources)


@app.post("/ingest")
async def ingest_document(req: Request, file: UploadFile = File(...)):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Queueing document ingestion", extra={"trace_id": trace_id, "filename": file.filename})

    content = await file.read()
    metadata = {"filename": file.filename, "content_type": file.content_type}
    task = ingest_document_task.delay(content.decode("utf-8", errors="ignore"), metadata)

    return {"status": "accepted", "task_id": str(task.id), "filename": file.filename}


@app.get("/ingest/status/{task_id}")
async def ingest_status(task_id: str):
    from .worker import celery_app
    result = celery_app.AsyncResult(task_id)
    return {"task_id": task_id, "status": result.status, "result": result.result if result.ready() else None}


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag-service"}


@app.get("/readiness")
def readiness():
    if llm_ready():
        return {"ready": True}
    raise HTTPException(status_code=503, detail="RAG models not loaded")
