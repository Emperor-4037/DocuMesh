import os
import uuid
import time
import httpx
from fastapi import FastAPI, Depends, Request, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from shared.config import settings
from shared.logging import setup_logging
from shared.auth import verify_token
from shared.metrics import instrument_app, REQUEST_COUNT, LatencyTracker
from shared.db import AsyncSessionLocal
from shared.models import AuditLog
from shared.schemas import (
    ParaphraseRequest, ParaphraseResponse,
    GrammarRequest, GrammarResponse,
    SimplifyRequest, SimplifyResponse,
    ToneRequest, ToneResponse,
    SummarizeRequest, SummarizeResponse,
    RAGQueryRequest, RAGQueryResponse,
)

# ── Service URL config (env-overridable for local dev) ────────────────────────
_PARAPHRASE_BASE = os.environ.get("PARAPHRASE_SERVICE_URL", "http://paraphrase-service:8000")
_GRAMMAR_BASE    = os.environ.get("GRAMMAR_SERVICE_URL",    "http://grammar-service:8000")
_SIMPLIFY_BASE   = os.environ.get("SIMPLIFY_SERVICE_URL",   "http://simplify-service:8000")
_TONE_BASE       = os.environ.get("TONE_SERVICE_URL",        "http://tone-service:8000")
_SUMMARIZE_BASE  = os.environ.get("SUMMARIZE_SERVICE_URL",  "http://summarize-service:8000")
_RAG_BASE        = os.environ.get("RAG_SERVICE_URL",         "http://rag-service:8000")

logger = setup_logging("gateway")
http_client = httpx.AsyncClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup (idempotent)
    try:
        from shared.db import engine, Base
        from shared.models import AuditLog  # noqa: F401 — ensure model is registered
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created/verified")
    except Exception as exc:
        logger.warning(f"DB init skipped: {exc}")
    yield
    await http_client.aclose()


app = FastAPI(title="AI Platform Gateway", version="1.0.0", lifespan=lifespan)
instrument_app(app, "gateway")

# Allow the frontend container (and local dev) to call the gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: trace ID + audit log ──────────────────────────────────────────
@app.middleware("http")
async def trace_and_audit(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
    request.state.trace_id = trace_id
    start = time.monotonic()

    logger.info("Incoming request", extra={
        "trace_id": trace_id, "path": request.url.path, "method": request.method
    })

    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Duration-Ms"] = str(duration_ms)

    # Fire-and-forget audit log (skip health / metrics endpoints)
    if not request.url.path.startswith(("/health", "/metrics")):
        try:
            async with AsyncSessionLocal() as session:
                user_id = getattr(request.state, "user_id", None)
                log = AuditLog(
                    trace_id=trace_id,
                    user_id=user_id,
                    method=request.method,
                    path=str(request.url.path),
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                )
                session.add(log)
                await session.commit()
        except Exception as exc:
            # Never let audit failure break the request
            logger.warning("Audit log write failed", extra={"error": str(exc)})

    return response


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "gateway"}


@app.get("/readiness")
async def readiness_check():
    """Check if all downstream services are reachable."""
    services = {
        "paraphrase": f"{_PARAPHRASE_BASE}/health",
        "grammar":    f"{_GRAMMAR_BASE}/health",
        "simplify":   f"{_SIMPLIFY_BASE}/health",
        "tone":       f"{_TONE_BASE}/health",
        "summarize":  f"{_SUMMARIZE_BASE}/health",
        "rag":        f"{_RAG_BASE}/health",
    }
    results = {}
    all_ok = True
    for name, url in services.items():
        try:
            resp = await http_client.get(url, timeout=3.0)
            results[name] = {"status": "ok", "code": resp.status_code}
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            all_ok = False

    status_code = 200 if all_ok else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={"ready": all_ok, "services": results},
        status_code=status_code,
    )


@app.post("/api/paraphrase", response_model=ParaphraseResponse)
async def paraphrase(request: ParaphraseRequest, req: Request, user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    with LatencyTracker("gateway", "/api/paraphrase"):
        result = await call_downstream(f"{_PARAPHRASE_BASE}/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/paraphrase", status_code=200).inc()
    return result


@app.post("/api/grammar", response_model=GrammarResponse)
async def grammar(request: GrammarRequest, req: Request, user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    with LatencyTracker("gateway", "/api/grammar"):
        result = await call_downstream(f"{_GRAMMAR_BASE}/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/grammar", status_code=200).inc()
    return result


@app.post("/api/simplify", response_model=SimplifyResponse)
async def simplify(request: SimplifyRequest, req: Request, user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    with LatencyTracker("gateway", "/api/simplify"):
        result = await call_downstream(f"{_SIMPLIFY_BASE}/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/simplify", status_code=200).inc()
    return result


@app.post("/api/tone", response_model=ToneResponse)
async def tone(request: ToneRequest, req: Request, user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    with LatencyTracker("gateway", "/api/tone"):
        result = await call_downstream(f"{_TONE_BASE}/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/tone", status_code=200).inc()
    return result


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest, req: Request, user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    with LatencyTracker("gateway", "/api/summarize"):
        result = await call_downstream(f"{_SUMMARIZE_BASE}/process", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/summarize", status_code=200).inc()
    return result


@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest, req: Request, user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    with LatencyTracker("gateway", "/api/rag/query"):
        result = await call_downstream(f"{_RAG_BASE}/query", request.dict(), req.state.trace_id)
    REQUEST_COUNT.labels(service="gateway", endpoint="/api/rag/query", status_code=200).inc()
    return result


@app.post("/api/rag/ingest")
async def rag_ingest(req: Request, file: UploadFile = File(...), user_id: str = Depends(verify_token)):
    req.state.user_id = user_id
    trace_id = req.state.trace_id
    files = {"file": (file.filename, await file.read(), file.content_type)}
    try:
        resp = await http_client.post(
            f"{_RAG_BASE}/ingest",
            files=files,
            headers={"X-Trace-Id": trace_id},
            timeout=60.0,
        )
        resp.raise_for_status()
        REQUEST_COUNT.labels(service="gateway", endpoint="/api/rag/ingest", status_code=200).inc()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error("Downstream error on ingest", extra={"trace_id": trace_id, "status": e.response.status_code})
        raise HTTPException(status_code=e.response.status_code, detail=f"Ingestion failed: {e.response.text}")
    except Exception as e:
        logger.error("Ingestion failed", extra={"trace_id": trace_id, "error": str(e)})
        raise HTTPException(status_code=500, detail="Internal ingestion error")


@app.get("/api/rag/ingest/status/{task_id}")
async def rag_ingest_status(task_id: str, req: Request, user_id: str = Depends(verify_token)):
    try:
        resp = await http_client.get(
            f"{_RAG_BASE}/ingest/status/{task_id}",
            headers={"X-Trace-Id": req.state.trace_id},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
