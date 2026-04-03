"""
RAG LLM answer generation using TinyLlama-1.1B-Chat.
  - 1.1B params, runs on consumer GPU with 4-bit quantization (<2GB VRAM)
  - Falls back cleanly to CPU if no GPU available
  - Swap MODEL_NAME for Mistral-7B / Llama-3.1-8B for higher quality
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from shared.model_utils import DEVICE
from shared.logging import setup_logging

# BitsAndBytes is optional — only available when CUDA is present
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

logger = setup_logging("rag-service")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_tokenizer = None
_model = None


def _build_bnb_config():
    """4-bit quantization via bitsandbytes (CUDA only)."""
    if DEVICE != "cuda" or not _HAS_BNB:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def _load():
    global _tokenizer, _model
    if _model is None:
        logger.info(f"Loading RAG LLM '{MODEL_NAME}' on {DEVICE}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        bnb = _build_bnb_config()

        if bnb:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb,
                device_map="auto",
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
            ).to(DEVICE)

        _model.eval()
        logger.info("RAG LLM ready.")


_SYSTEM_PROMPT = (
    "You are an accurate, helpful assistant. "
    "Answer questions using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Be concise and factual."
)


def generate_answer(query: str, passages: list[dict], max_new_tokens: int = 300) -> str:
    _load()

    # Build context block from retrieved passages
    context = "\n\n".join(
        f"[Source: {p.get('source', 'unknown')}]\n{p['text']}"
        for p in passages[:5]
    )

    # TinyLlama uses ChatML format
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer based only on the context above:"
            ),
        },
    ]

    # Apply chat template
    prompt = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                # greedy for factual tasks
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
