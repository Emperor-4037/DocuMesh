"""
Shared adapter-aware model loader.
All service model.py files call load_model_with_adapter() to automatically
use a fine-tuned LoRA adapter if one is present in ./models/{service_name}/.
"""
from pathlib import Path
from shared.logging import setup_logging

logger = setup_logging("adapter_loader")

ADAPTERS_ROOT = Path(__file__).parent.parent / "models"


def get_adapter_path(service_name: str) -> str | None:
    """Return the adapter directory path if a fine-tuned adapter exists, else None."""
    adapter_dir = ADAPTERS_ROOT / service_name
    if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
        logger.info(f"Found fine-tuned adapter for '{service_name}' at {adapter_dir}")
        return str(adapter_dir)
    return None


def load_seq2seq_model(base_model_name: str, service_name: str, dtype, device: str):
    """
    Load a Seq2Seq model. If a LoRA adapter exists, merge it with the base model
    for seamless inference (no code changes needed downstream).
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    adapter_path = get_adapter_path(service_name)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model_name, legacy=False)

    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading base model + LoRA adapter for {service_name}...")
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, adapter_path)
        # Merge weights for fast inference (removes adapter overhead)
        model = model.merge_and_unload()
    else:
        logger.info(f"Loading base model (no adapter) for {service_name}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)

    return tokenizer, model.to(device).eval()
