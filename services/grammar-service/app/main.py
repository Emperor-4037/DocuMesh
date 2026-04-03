from fastapi import FastAPI, Request
from shared.schemas import GrammarRequest, GrammarResponse, Correction
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import correct_grammar

logger = setup_logging("grammar-service")
app = FastAPI(title="Grammar Service")
instrument_app(app, "grammar-service")


@app.post("/process", response_model=GrammarResponse)
async def process_grammar(request: GrammarRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing grammar request", extra={"trace_id": trace_id, "text_length": len(request.text)})

    with LatencyTracker("grammar-service", "/process"):
        corrected_text, raw_corrections = correct_grammar(request.text)

    corrections = [
        Correction(
            original=c["original"],
            replacement=c["replacement"],
            start_index=c["start_index"],
            end_index=c["end_index"],
            description=c["description"],
        )
        for c in raw_corrections
    ]
    return GrammarResponse(corrected_text=corrected_text, corrections=corrections)


@app.get("/health")
def health():
    return {"status": "ok"}
