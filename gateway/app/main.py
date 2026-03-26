import uuid
import httpx
from fastapi import FastAPI, Depends, Request, HTTPException
from contextlib import asynccontextmanager

from shared.config import settings
from shared.logging import setup_logging
from shared.auth import verify_token
from shared.metrics import instrument_app, REQUEST_COUNT, LatencyTracker
from shared.schemas import (
    ParaphraseRequest, ParaphraseResponse,
    GrammarRequest, GrammarResponse,
    SimplifyRequest, SimplifyResponse,
    ToneRequest, ToneResponse,
    SummarizeRequest, SummarizeResponse,
    RAGQueryRequest, RAGQueryResponse,
)

logger = setup_logging("gateway")
http_client = httpx.AsyncClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await http_client.aclose()

app = FastAPI(title="AI Platform Gateway", version="1.0.0", lifespan=lifespan)
instrument_app(app, "gateway")

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    logger.info("Incoming request", extra={"trace_id": trace_id, "path": request.url.path, "method": request.method})
    response = await call_next(request)
    response.headers["X-Trace-Id"] = trace_id
    return response

async def call_downstream(service_url: str, payload: dict, trace_id: str):
    headers = {"X-Trace-Id": trace_id, "Content-Type": "application/json"}
    try:
        resp = await http_client.post(service_url, json=payload, headers=headers, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("Downstream error", extra={"trace_id": trace_id, "status": e.response.status_code})
        raise HTTPException(status_code=e.response.status_code, detail=f"Downstream service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error("Downstream unreachable", extra={"trace_id": trace_id, "error": str(e)})
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gateway"}

@app.post("/api/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(request: ParaphraseRequest, req: Request, user_id: str = Depends(verify_token)):
    with LatencyTracker("gateway", "/api/paraphrase"):
        result = await call_downstream("http://paraphrase-service:8000/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/paraphrase", status_code=200).inc()
    return result

@app.post("/api/grammar", response_model=GrammarResponse)
async def grammar(request: GrammarRequest, req: Request, user_id: str = Depends(verify_token)):
    with LatencyTracker("gateway", "/api/grammar"):
        result = await call_downstream("http://grammar-service:8000/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/grammar", status_code=200).inc()
    return result

@app.post("/api/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest, req: Request, user_id: str = Depends(verify_token)):
    with LatencyTracker("gateway", "/api/simplify"):
        result = await call_downstream("http://simplify-service:8000/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/simplify", status_code=200).inc()
    return result

@app.post("/api/tone", response_model=ToneResponse)
async def tone(request: ToneRequest, req: Request, user_id: str = Depends(verify_token)):
    with LatencyTracker("gateway", "/api/tone"):
        result = await call_downstream("http://tone-service:8000/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/tone", status_code=200).inc()
    return result

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest, req: Request, user_id: str = Depends(verify_token)):
    with LatencyTracker("gateway", "/api/summarize"):
        result = await call_downstream("http://summarize-service:8000/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/summarize", status_code=200).inc()
    return result

@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest, req: Request, user_id: str = Depends(verify_token)):
    with LatencyTracker("gateway", "/api/rag/query"):
        result = await call_downstream("http://rag-service:8000/query", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/rag/query", status_code=200).inc()
    return result

@app.post("/api/rag/ingest")
async def rag_ingest(req: Request, user_id: str = Depends(verify_token)):
    result = await call_downstream("http://rag-service:8000/ingest", {}, req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/rag/ingest", status_code=200).inc()
    return result
