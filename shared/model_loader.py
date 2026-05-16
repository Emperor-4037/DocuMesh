"""
Config-driven model loader with support for:
  - Seq2Seq models (T5, BART) with optional LoRA adapters
  - Causal LLMs (TinyLlama, Llama, Mistral) with optional 4-bit quantization
  - HuggingFace pipelines (summarization)
  - Startup validation (ensures model is loadable before serving)

Usage:
    from shared.model_loader import load_model
    from shared.model_config import PARAPHRASE_CONFIG
    tokenizer, model = load_model(PARAPHRASE_CONFIG)
"""
import logging
from pathlib import Path
from typing import Tuple, Any

import torch
from transformers import AutoTokenizer

from .model_config import ModelConfig
from .model_utils import DEVICE, DTYPE

logger = logging.getLogger("model_loader")


def load_model(config: ModelConfig) -> Tuple[Any, Any]:
    """
    Load a model + tokenizer based on ModelConfig.
    Returns (tokenizer, model) ready for inference.

    Supports:
      model_type="seq2seq"  → AutoModelForSeq2SeqLM + optional LoRA merge
      model_type="causal"   → AutoModelForCausalLM + optional 4-bit quantization
      model_type="pipeline" → HuggingFace pipeline (returns None, pipeline)
    """
    device = config.device or DEVICE
    dtype = DTYPE

    if config.model_type == "seq2seq":
        return _load_seq2seq(config, device, dtype)
    elif config.model_type == "causal":
        return _load_causal(config, device)
    elif config.model_type == "pipeline":
        return _load_pipeline(config, device, dtype)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")


def _resolve_adapter(config: ModelConfig) -> str | None:
    """Check if adapter path exists and has valid files."""
    if not config.adapter_path:
        return None
    p = Path(config.adapter_path)
    if p.exists() and (p / "adapter_config.json").exists():
        logger.info(f"Found adapter at {p}")
        return str(p)
    logger.info(f"No adapter found at {p} — using base model only")
    return None


def _load_seq2seq(config: ModelConfig, device: str, dtype: torch.dtype):
    """Load a Seq2Seq model (T5, BART) with optional LoRA adapter."""
    from transformers import AutoModelForSeq2SeqLM

    adapter_path = _resolve_adapter(config)
    model_source = adapter_path or config.base_model

    logger.info(f"Loading seq2seq: base={config.base_model}, adapter={adapter_path}, device={device}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, legacy=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model, dtype=dtype)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        logger.info("Merged LoRA adapter into base model")
    else:
        model = base_model

    model = model.to(device).eval()
    logger.info(f"Seq2Seq model ready on {device}")
    return tokenizer, model


def _load_causal(config: ModelConfig, device: str):
    """Load a causal LLM with optional 4-bit quantization and LoRA adapter."""
    from transformers import AutoModelForCausalLM

    adapter_path = _resolve_adapter(config)

    logger.info(f"Loading causal: base={config.base_model}, 4bit={config.quantize_4bit}, device={device}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    # 4-bit quantization (CUDA only)
    bnb_config = None
    if config.quantize_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except ImportError:
            logger.warning("bitsandbytes not available — loading without quantization")

    if bnb_config:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        ).to(device)

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Applied LoRA adapter to causal model")

    model.eval()
    logger.info(f"Causal model ready on {device}")
    return tokenizer, model


def _load_pipeline(config: ModelConfig, device: str, dtype: torch.dtype):
    """Load a HuggingFace pipeline (e.g., for summarization)."""
    from transformers import pipeline

    logger.info(f"Loading pipeline: model={config.base_model}, device={device}")

    pipe = pipeline(
        "text2text-generation",
        model=config.base_model,
        tokenizer=config.base_model,
        device=0 if device == "cuda" else -1,
        dtype=dtype,
    )
    logger.info("Pipeline ready")
    return None, pipe


def warmup(tokenizer, model, config: ModelConfig) -> bool:
    """
    Run a dummy inference pass to warm up the model.
    Returns True if warmup succeeds, False otherwise.
    Call this during startup before marking service as ready.
    """
    try:
        if config.model_type in ("seq2seq", "causal"):
            dummy_input = "This is a warmup test."
            device = config.device or DEVICE
            inputs = tokenizer(dummy_input, return_tensors="pt", truncation=True, max_length=32)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=10)
            logger.info("Warmup pass completed successfully")
            return True
        elif config.model_type == "pipeline":
            model("This is a warmup test.", max_length=20, truncation=True)
            logger.info("Pipeline warmup completed successfully")
            return True
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
    return False
