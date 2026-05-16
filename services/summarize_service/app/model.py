"""
Summarization model — config-driven, loaded once at startup.
Uses HuggingFace pipeline for BART-Large-CNN.
Supports hierarchical chunked summarization for long documents.
"""
from shared.model_config import SUMMARIZE_CONFIG
from shared.model_loader import load_model, warmup
from shared.model_utils import DEVICE, DTYPE
from shared.logging import setup_logging

logger = setup_logging("summarize-service")

_pipeline = None
_ready = False


def _load():
    global _pipeline, _ready
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline

        cfg = SUMMARIZE_CONFIG
        logger.info(f"Loading summarize model '{cfg.base_model}' on {DEVICE}...")
        _pipeline = hf_pipeline(
            "text2text-generation",
            model=cfg.base_model,
            tokenizer=cfg.base_model,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=DTYPE,
        )
        # Warmup
        try:
            _pipeline("This is a warmup test for summarization.", max_length=20, truncation=True)
            _ready = True
            logger.info("Summarize model ready.")
        except Exception as e:
            logger.error(f"Summarize warmup failed: {e}")


def is_ready() -> bool:
    return _ready


def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    _load()

    if len(text) <= 4000:
        result = _pipeline(
            text,
            max_length=max(max_length, 50),
            min_length=min(min_length, max_length - 10),
            do_sample=False,
            truncation=True,
        )
        return result[0]["generated_text"]

    # Hierarchical summarization for long documents
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    chunk_summaries = []
    for chunk in chunks:
        out = _pipeline(chunk, max_length=150, min_length=30, do_sample=False, truncation=True)
        chunk_summaries.append(out[0]["generated_text"])

    combined = " ".join(chunk_summaries)
    final = _pipeline(combined, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
    return final[0]["generated_text"]
