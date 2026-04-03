from fastapi import FastAPI, Request
from shared.schemas import SimplifyRequest, SimplifyResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import simplify

logger = setup_logging("simplify-service")
app = FastAPI(title="Simplify Service")
instrument_app(app, "simplify-service")


@app.post("/process", response_model=SimplifyResponse)
async def process_simplify(request: SimplifyRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing simplify request", extra={"trace_id": trace_id, "level": request.reading_level})

    with LatencyTracker("simplify-service", "/process"):
        result = simplify(request.text, reading_level=request.reading_level or "basic")

    return SimplifyResponse(simplified_text=result)


@app.get("/health")
def health():
    return {"status": "ok"}
