from fastapi import FastAPI, Request
from shared.schemas import GrammarRequest, GrammarResponse, Correction
from shared.logging import setup_logging

logger = setup_logging("grammar-service")
app = FastAPI(title="Grammar Service")

@app.post("/process", response_model=GrammarResponse)
async def process_grammar(request: GrammarRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing grammar request", extra={"trace_id": trace_id, "text_length": len(request.text)})
    
    # MVP Placeholder for grammar check model execution
    corrected = f"[Corrected] {request.text}"
    corrections = [
        Correction(original="bad", replacement="good", start_index=0, end_index=3, description="Fixed grammar")
    ]
    
    return GrammarResponse(corrected_text=corrected, corrections=corrections)

@app.get("/health")
def health():
    return {"status": "ok"}
