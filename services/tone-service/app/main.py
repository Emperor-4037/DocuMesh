from fastapi import FastAPI, Request
from shared.schemas import ToneRequest, ToneResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import transfer_tone

logger = setup_logging("tone-service")
app = FastAPI(title="Tone Service")
instrument_app(app, "tone-service")


@app.post("/process", response_model=ToneResponse)
async def process_tone(request: ToneRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing tone request", extra={"trace_id": trace_id, "target_tone": request.target_tone})

    with LatencyTracker("tone-service", "/process"):
        result = transfer_tone(request.text, target_tone=request.target_tone)

    return ToneResponse(toned_text=result)


@app.get("/health")
def health():
    return {"status": "ok"}
