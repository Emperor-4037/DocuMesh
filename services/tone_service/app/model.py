"""
Tone / style transfer model — config-driven, loaded once at startup.
"""
import torch
from shared.model_config import TONE_CONFIG
from shared.model_loader import load_model, warmup
from shared.model_utils import DEVICE
from shared.logging import setup_logging

logger = setup_logging("tone-service")

_tokenizer = None
_model = None
_ready = False

_TONE_PROMPTS = {
    "formal":       "Rewrite the following text in a formal, professional tone suitable for business communication:\n\n",
    "informal":     "Rewrite the following text in a friendly, informal conversational tone:\n\n",
    "professional": "Rewrite the following text to sound polished and professional, as if written by a senior executive:\n\n",
    "casual":       "Rewrite the following text in a relaxed, casual tone like chatting with a friend:\n\n",
    "assertive":    "Rewrite the following text to be confident and assertive, making strong clear statements:\n\n",
    "empathetic":   "Rewrite the following text in a warm, empathetic, and understanding tone:\n\n",
    "persuasive":   "Rewrite the following text to be highly persuasive and compelling, motivating the reader to act:\n\n",
    "academic":     "Rewrite the following text in an academic, scholarly tone with precise language:\n\n",
}


def _load():
    global _tokenizer, _model, _ready
    if _model is None:
        _tokenizer, _model = load_model(TONE_CONFIG)
        _ready = warmup(_tokenizer, _model, TONE_CONFIG)


def is_ready() -> bool:
    return _ready


def transfer_tone(text: str, target_tone: str = "professional") -> str:
    _load()
    cfg = TONE_CONFIG

    prompt = _TONE_PROMPTS.get(
        target_tone.lower(),
        f"Rewrite the following text in a {target_tone} tone:\n\n"
    )
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
