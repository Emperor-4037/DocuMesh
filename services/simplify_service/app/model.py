"""
Text simplification model — config-driven, loaded once at startup.
"""
import torch
from shared.model_config import SIMPLIFY_CONFIG
from shared.model_loader import load_model, warmup
from shared.model_utils import DEVICE
from shared.logging import setup_logging

logger = setup_logging("simplify-service")

_tokenizer = None
_model = None
_ready = False

_LEVEL_PROMPTS = {
    "basic":        "Rewrite the following text in very simple words that a 10-year-old child can understand:\n\n",
    "intermediate": "Rewrite the following text in plain English, avoiding jargon and complex vocabulary:\n\n",
    "advanced":     "Simplify the following text while keeping technical accuracy. Remove unnecessary complexity:\n\n",
}


def _load():
    global _tokenizer, _model, _ready
    if _model is None:
        _tokenizer, _model = load_model(SIMPLIFY_CONFIG)
        _ready = warmup(_tokenizer, _model, SIMPLIFY_CONFIG)


def is_ready() -> bool:
    return _ready


def simplify(text: str, reading_level: str = "basic") -> str:
    _load()
    cfg = SIMPLIFY_CONFIG

    prompt = _LEVEL_PROMPTS.get(reading_level.lower(), _LEVEL_PROMPTS["basic"])
    inputs = _tokenizer(
        prompt + text,
        return_tensors="pt",
        max_length=cfg.max_input_tokens,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=cfg.max_output_tokens,
            num_beams=cfg.num_beams,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            early_stopping=True,
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
