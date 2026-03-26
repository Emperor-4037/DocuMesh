from fastapi import FastAPI, Request
from shared.schemas import SummarizeRequest, SummarizeResponse
from shared.logging import setup_logging

logger = setup_logging("summarize-service")
app = FastAPI(title="Summarize Service")

@app.post("/process", response_model=SummarizeResponse)
async def process_summarize(request: SummarizeRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing summarize request", extra={"trace_id": trace_id, "text_length": len(request.text)})
    
    # MVP Placeholder
    summary_len = min(len(request.text), request.max_length) if request.max_length else len(request.text)
    summary = f"[Summarized] {request.text[:summary_len]}..."
    
    return SummarizeResponse(summary=summary)

@app.get("/health")
def health():
    return {"status": "ok"}
