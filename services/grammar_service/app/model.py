"""
Grammar correction model — config-driven, loaded once at startup.
"""
import difflib
import torch
from shared.model_config import GRAMMAR_CONFIG
from shared.model_loader import load_model, warmup
from shared.model_utils import DEVICE
from shared.logging import setup_logging

logger = setup_logging("grammar-service")

_tokenizer = None
_model = None
_ready = False


def _load():
    global _tokenizer, _model, _ready
    if _model is None:
        _tokenizer, _model = load_model(GRAMMAR_CONFIG)
        _ready = warmup(_tokenizer, _model, GRAMMAR_CONFIG)


def is_ready() -> bool:
    return _ready


def correct_grammar(text: str) -> tuple[str, list[dict]]:
    """Returns (corrected_text, corrections_list)."""
    _load()
    cfg = GRAMMAR_CONFIG

    inputs = _tokenizer(
        "grammar: " + text,
        return_tensors="pt",
        max_length=cfg.max_input_tokens,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=cfg.max_output_tokens,
            num_beams=cfg.num_beams,
            early_stopping=True,
        )

    corrected = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    corrections = _build_corrections(text, corrected)
    return corrected, corrections


def _build_corrections(original: str, corrected: str) -> list[dict]:
    """Token-level diff between original and corrected text."""
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    results = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_span = " ".join(orig_words[i1:i2])
        corr_span = " ".join(corr_words[j1:j2])
        start = sum(len(w) + 1 for w in orig_words[:i1])
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
