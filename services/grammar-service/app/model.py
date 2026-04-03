"""
Grammar correction model loader and inference.
Model: vennify/t5-base-grammar-correction
  - T5-base fine-tuned on C4 grammar correction dataset (248M params)
  - Returns corrected text and a simple diff of changes
"""
import difflib
import torch
from shared.model_utils import DEVICE, DTYPE
from shared.adapter_loader import load_seq2seq_model
from shared.logging import setup_logging

logger = setup_logging("grammar-service")

MODEL_NAME = "vennify/t5-base-grammar-correction"
SERVICE_NAME = "grammar"

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer, _model = load_seq2seq_model(MODEL_NAME, SERVICE_NAME, DTYPE, DEVICE)


def correct_grammar(text: str) -> tuple[str, list[dict]]:
    """
    Returns (corrected_text, corrections) where corrections is a list of
    {original, replacement, start_index, end_index, description} dicts.
    """
    _load()

    prefix = "grammar: "
    inputs = _tokenizer(
        prefix + text, return_tensors="pt", max_length=512, truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True,
        )

    corrected = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Build a lightweight diff between original and corrected
    corrections = _build_corrections(text, corrected)
    return corrected, corrections


def _build_corrections(original: str, corrected: str) -> list[dict]:
    """Use SequenceMatcher to produce token-level correction records."""
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    results = []
    char_offset = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_span = " ".join(orig_words[i1:i2])
        corr_span = " ".join(corr_words[j1:j2])
        start = sum(len(w) + 1 for w in orig_words[:i1])  # rough char offset
        end = start + len(orig_span)

        if tag in ("replace", "delete", "insert") and orig_span != corr_span:
            results.append({
                "original": orig_span,
                "replacement": corr_span,
                "start_index": start,
                "end_index": end,
                "description": f"{tag.capitalize()}: '{orig_span}' → '{corr_span}'"
            })

    return results
