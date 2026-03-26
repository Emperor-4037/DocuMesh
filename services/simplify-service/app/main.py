from fastapi import FastAPI, Request
from shared.schemas import SimplifyRequest, SimplifyResponse
from shared.logging import setup_logging

logger = setup_logging("simplify-service")
app = FastAPI(title="Simplify Service")

@app.post("/process", response_model=SimplifyResponse)
async def process_simplify(request: SimplifyRequest, req: Request):
    trace_id = req.headers.get("X-Trace-Id", "unknown")
    logger.info("Processing simplify request", extra={"trace_id": trace_id, "text_length": len(request.text)})
    
    # MVP Placeholder for actual model inference
    simplified = f"[Simplified to {request.reading_level} level] {request.text}"
    return SimplifyResponse(simplified_text=simplified)

@app.get("/health")
def health():
    return {"status": "ok"}
