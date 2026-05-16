import asyncio
from fastapi import FastAPI, Request, HTTPException
from shared.schemas import ParaphraseRequest, ParaphraseResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import paraphrase, is_ready, _load

logger = setup_logging("paraphrase-service")
app = FastAPI(title="Paraphrase Service")
instrument_app(app, "paraphrase-service")


@app.on_event("startup")
async def startup():
    """Load model on startup so first request is fast."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load)
    logger.info("Paraphrase service started and model loaded")


@app.post("/process", response_model=ParaphraseResponse)
async def process_paraphrase(request: ParaphraseRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing paraphrase request", extra={"trace_id": trace_id, "text_length": len(request.text)})

    loop = asyncio.get_event_loop()
    with LatencyTracker("paraphrase-service", "/process"):
        result = await loop.run_in_executor(None, paraphrase, request.text, request.tone or "neutral")

    return ParaphraseResponse(paraphrased_text=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "paraphrase-service"}


@app.get("/readiness")
def readiness():
    if is_ready():
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Model not loaded")
