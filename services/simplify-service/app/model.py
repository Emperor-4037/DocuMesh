"""
Text simplification model.
Model: google/flan-t5-base (instruction-prompted simplification)
  - Flan-T5 reliably follows "simplify" instructions
  - Supports reading level targeting via prompt engineering
"""
import torch
from shared.model_utils import DEVICE, DTYPE
from shared.adapter_loader import load_seq2seq_model
from shared.logging import setup_logging

logger = setup_logging("simplify-service")

MODEL_NAME = "google/flan-t5-base"
SERVICE_NAME = "simplify"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = load_seq2seq_model(MODEL_NAME, SERVICE_NAME, DTYPE, DEVICE)


_LEVEL_PROMPTS = {
    "basic":        "Rewrite the following text in very simple words that a 10-year-old child can understand:\n\n",
    "intermediate": "Rewrite the following text in plain English, avoiding jargon and complex vocabulary:\n\n",
    "advanced":     "Simplify the following text while keeping technical accuracy. Remove unnecessary complexity:\n\n",
}


def simplify(text: str, reading_level: str = "basic") -> str:
    _load()

    prompt = _LEVEL_PROMPTS.get(reading_level.lower(), _LEVEL_PROMPTS["basic"])
    input_text = prompt + text

    inputs = _tokenizer(input_text, return_tensors="pt", max_length=768, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
