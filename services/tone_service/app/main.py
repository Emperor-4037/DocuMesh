import asyncio
from fastapi import FastAPI, Request, HTTPException
from shared.schemas import ToneRequest, ToneResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import transfer_tone, is_ready, _load

logger = setup_logging("tone-service")
app = FastAPI(title="Tone Service")
instrument_app(app, "tone-service")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load)
    logger.info("Tone service started and model loaded")


@app.post("/process", response_model=ToneResponse)
async def process_tone(request: ToneRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing tone request", extra={"trace_id": trace_id, "target_tone": request.target_tone})

    loop = asyncio.get_event_loop()
    with LatencyTracker("tone-service", "/process"):
        result = await loop.run_in_executor(None, transfer_tone, request.text, request.target_tone)

    return ToneResponse(toned_text=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "tone-service"}


@app.get("/readiness")
def readiness():
    if is_ready():
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Model not loaded")
