from fastapi import FastAPI, Request
from shared.schemas import ToneRequest, ToneResponse
from shared.logging import setup_logging

logger = setup_logging("tone-service")
app = FastAPI(title="Tone Service")

@app.post("/process", response_model=ToneResponse)
async def process_tone(request: ToneRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing tone transfer request", extra={"trace_id": trace_id, "target_tone": request.target_tone})
    
    # MVP Placeholder
    toned = f"[Tone applied: {request.target_tone}] {request.text}"
    return ToneResponse(toned_text=toned)

@app.get("/health")
def health():
    return {"status": "ok"}
