"""
Paraphrase model loader and inference.
Model: humarin/chatgpt_paraphraser_on_T5_base
  - Fine-tuned T5-base for paraphrasing (248M params)
  - Loaded once at startup (singleton) to avoid per-request overhead
"""
import torch
from shared.model_utils import DEVICE, DTYPE
from shared.adapter_loader import load_seq2seq_model
from shared.logging import setup_logging

logger = setup_logging("paraphrase-service")

MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"
SERVICE_NAME = "paraphrase"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = load_seq2seq_model(MODEL_NAME, SERVICE_NAME, DTYPE, DEVICE)


def paraphrase(text: str, tone: str = "neutral", num_beams: int = 5, num_return_sequences: int = 1) -> str:
    _load()

    # Prepend tone instruction for different styles
    tone_prefix = {
        "formal":    "paraphrase formally: ",
        "casual":    "paraphrase casually: ",
        "creative":  "paraphrase creatively: ",
        "concise":   "paraphrase concisely: ",
        "neutral":   "paraphrase: ",
    }.get(tone.lower(), "paraphrase: ")

    input_text = tone_prefix + text
    inputs = _tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=3,
            max_length=512,
            early_stopping=True,
        )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True)
