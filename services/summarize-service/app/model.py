"""
Summarization model loader and inference.
Model: facebook/bart-large-cnn
  - BART-Large fine-tuned on CNN/DailyMail (406M params)
  - Industry-standard abstractive summarizer; handles long documents well
"""
import torch
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from shared.model_utils import DEVICE, DTYPE
from shared.logging import setup_logging

logger = setup_logging("summarize-service")

MODEL_NAME = "facebook/bart-large-cnn"

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        logger.info(f"Loading summarize model '{MODEL_NAME}' on {DEVICE}...")
        _pipeline = pipeline(
            "summarization",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=DTYPE,
        )
        logger.info("Summarize model ready.")


def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    _load()

    # BART input limit is 1024 tokens; chunk long documents
    chunk_size = 900  # chars (conservative)
    if len(text) <= 4000:
        # Single pass
        result = _pipeline(
            text,
            max_length=max(max_length, 50),
            min_length=min(min_length, max_length - 10),
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"]

    # Multi-chunk: summarize chunks then summarize the summaries
    chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
    chunk_summaries = []
    for chunk in chunks:
        out = _pipeline(chunk, max_length=150, min_length=30, do_sample=False, truncation=True)
        chunk_summaries.append(out[0]["summary_text"])

    combined = " ".join(chunk_summaries)
    final = _pipeline(combined, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
    return final[0]["summary_text"]
