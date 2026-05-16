import asyncio
from fastapi import FastAPI, Request, HTTPException
from shared.schemas import SummarizeRequest, SummarizeResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import summarize, is_ready, _load

logger = setup_logging("summarize-service")
app = FastAPI(title="Summarize Service")
instrument_app(app, "summarize-service")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load)
    logger.info("Summarize service started and model loaded")


@app.post("/process", response_model=SummarizeResponse)
async def process_summarize(request: SummarizeRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing summarize request", extra={"trace_id": trace_id, "text_length": len(request.text)})

    loop = asyncio.get_event_loop()
    with LatencyTracker("summarize-service", "/process"):
        result = await loop.run_in_executor(None, summarize, request.text, request.max_length or 150)

    return SummarizeResponse(summary=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "summarize-service"}


@app.get("/readiness")
def readiness():
    if is_ready():
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Model not loaded")
