"""
RAG LLM — config-driven causal model with 4-bit quantization support.
Uses TinyLlama-1.1B-Chat by default, configurable via MODEL_RAG_LLM_BASE env var.
"""
import torch
from shared.model_config import RAG_LLM_CONFIG
from shared.model_loader import load_model, warmup
from shared.logging import setup_logging

logger = setup_logging("rag-service")

_tokenizer = None
_model = None
_ready = False

_SYSTEM_PROMPT = (
    "You are an accurate, helpful assistant. "
    "Answer questions using ONLY the provided context. "
    "If the context does not contain enough information, say 'I cannot answer this based on the provided documents.' "
    "For each claim you make, cite the source in brackets like [Source: filename]. "
    "Be concise and factual."
)


def _load():
    global _tokenizer, _model, _ready
    if _model is None:
        _tokenizer, _model = load_model(RAG_LLM_CONFIG)
        _ready = warmup(_tokenizer, _model, RAG_LLM_CONFIG)


def is_ready() -> bool:
    return _ready


def generate_answer(query: str, passages: list[dict], max_new_tokens: int = 300) -> str:
    _load()

    # Build context from retrieved passages
    context = "\n\n".join(
        f"[Source: {p.get('source', 'unknown')}]\n{p['text']}"
        for p in passages[:5]
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer based only on the context above. Cite sources."
            ),
        },
    ]

    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=RAG_LLM_CONFIG.do_sample,
            temperature=RAG_LLM_CONFIG.temperature,
            repetition_penalty=RAG_LLM_CONFIG.repetition_penalty,
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
