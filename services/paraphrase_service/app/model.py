"""
Paraphrase model — config-driven, loaded once at startup.
"""
import torch
from shared.model_config import PARAPHRASE_CONFIG
from shared.model_loader import load_model, warmup
from shared.model_utils import DEVICE
from shared.logging import setup_logging

logger = setup_logging("paraphrase-service")

_tokenizer = None
_model = None
_ready = False

_TONE_PREFIX = {
    "formal":   "paraphrase formally: ",
    "casual":   "paraphrase casually: ",
    "creative": "paraphrase creatively: ",
    "concise":  "paraphrase concisely: ",
    "neutral":  "paraphrase: ",
}


def _load():
    global _tokenizer, _model, _ready
    if _model is None:
        _tokenizer, _model = load_model(PARAPHRASE_CONFIG)
        _ready = warmup(_tokenizer, _model, PARAPHRASE_CONFIG)


def is_ready() -> bool:
    return _ready


def paraphrase(text: str, tone: str = "neutral") -> str:
    _load()
    cfg = PARAPHRASE_CONFIG

    prefix = _TONE_PREFIX.get(tone.lower(), "paraphrase: ")
    inputs = _tokenizer(
        prefix + text,
        return_tensors="pt",
        max_length=cfg.max_input_tokens,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=cfg.max_output_tokens,
            num_beams=cfg.num_beams,
            no_repeat_ngram_size=cfg.no_repeat_ngram_size,
            early_stopping=True,
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
