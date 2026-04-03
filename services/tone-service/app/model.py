"""
Tone / style transfer model.
Model: google/flan-t5-base (instruction-prompted style transfer)
  - Supports: formal, informal, professional, casual, assertive, empathetic, persuasive
  - Uses detailed prompts to steer tone reliably
"""
import torch
from shared.model_utils import DEVICE, DTYPE
from shared.adapter_loader import load_seq2seq_model
from shared.logging import setup_logging

logger = setup_logging("tone-service")

MODEL_NAME = "google/flan-t5-base"
SERVICE_NAME = "tone"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = load_seq2seq_model(MODEL_NAME, SERVICE_NAME, DTYPE, DEVICE)


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


def transfer_tone(text: str, target_tone: str = "professional") -> str:
    _load()

    prompt = _TONE_PROMPTS.get(
        target_tone.lower(),
        f"Rewrite the following text in a {target_tone} tone:\n\n"
    )
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
