from fastapi import FastAPI, Request
from shared.schemas import SummarizeRequest, SummarizeResponse
from shared.logging import setup_logging
from shared.metrics import instrument_app, LatencyTracker
from .model import summarize

logger = setup_logging("summarize-service")
app = FastAPI(title="Summarize Service")
instrument_app(app, "summarize-service")


@app.post("/process", response_model=SummarizeResponse)
async def process_summarize(request: SummarizeRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing summarize request", extra={"trace_id": trace_id, "text_length": len(request.text)})

    with LatencyTracker("summarize-service", "/process"):
        result = summarize(request.text, max_length=request.max_length or 150)

    return SummarizeResponse(summary=result)


@app.get("/health")
def health():
    return {"status": "ok"}
