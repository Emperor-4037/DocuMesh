from fastapi import FastAPI, Request
from shared.schemas import ParaphraseRequest, ParaphraseResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import paraphrase

logger = setup_logging("paraphrase-service")
app = FastAPI(title="Paraphrase Service")
instrument_app(app, "paraphrase-service")


@app.post("/process", response_model=ParaphraseResponse)
async def process_paraphrase(request: ParaphraseRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing paraphrase request", extra={"trace_id": trace_id, "text_length": len(request.text)})

    with LatencyTracker("paraphrase-service", "/process"):
        result = paraphrase(request.text, tone=request.tone or "neutral")

    return ParaphraseResponse(paraphrased_text=result)


@app.get("/health")
def health():
    return {"status": "ok"}
