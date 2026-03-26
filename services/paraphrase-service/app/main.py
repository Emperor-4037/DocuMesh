from fastapi import FastAPI, Request
from shared.schemas import ParaphraseRequest, ParaphraseResponse
from shared.logging import setup_logging

logger = setup_logging("paraphrase-service")
app = FastAPI(title="Paraphrase Service")

@app.post("/process", response_model=ParaphraseResponse)
async def process_paraphrase(request: ParaphraseRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing paraphrase request", extra={"trace_id": trace_id, "text_length": len(request.text)})
    
    # MVP Placeholder: Replace with actual model inference (e.g. HuggingFace pipeline)
    paraphrased = f"[Paraphrased] {request.text}"
    if request.tone != "neutral":
        paraphrased += f" (in a {request.tone} tone)"
    
    return ParaphraseResponse(paraphrased_text=paraphrased)

@app.get("/health")
def health():
    return {"status": "ok"}
