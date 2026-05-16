import asyncio
from fastapi import FastAPI, Request, HTTPException
from shared.schemas import SimplifyRequest, SimplifyResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import simplify, is_ready, _load

logger = setup_logging("simplify-service")
app = FastAPI(title="Simplify Service")
instrument_app(app, "simplify-service")


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _load)
    logger.info("Simplify service started and model loaded")


@app.post("/process", response_model=SimplifyResponse)
async def process_simplify(request: SimplifyRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing simplify request", extra={"trace_id": trace_id, "level": request.reading_level})

    loop = asyncio.get_event_loop()
    with LatencyTracker("simplify-service", "/process"):
        result = await loop.run_in_executor(None, simplify, request.text, request.reading_level or "basic")

    return SimplifyResponse(simplified_text=result)


@app.get("/health")
def health():
    return {"status": "ok", "service": "simplify-service"}


@app.get("/readiness")
def readiness():
    if is_ready():
        return {"ready": True}
    raise HTTPException(status_code=503, detail="Model not loaded")
