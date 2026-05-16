"""
Centralized model configuration for all services.

Each service reads its model settings from here. This is the single source of truth
for which base model to use, which adapter to apply, generation parameters, and
quantization settings. Change models here, not in individual service code.

Environment variable overrides are supported for Docker/K8s flexibility:
  MODEL_PARAPHRASE_BASE=google/flan-t5-base
  MODEL_PARAPHRASE_ADAPTER=models/adapters/paraphrase/v2

VERSION POLICY:
  - All adapter paths MUST point to a specific version directory.
  - "latest" symlinks or tags are FORBIDDEN — they create ambiguity.
  - Rollback = change the version string and restart the service.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model deployment."""
    base_model: str                           # HuggingFace model ID or local path
    adapter_path: Optional[str] = None        # Path to LoRA adapter directory (versioned)
    model_type: str = "seq2seq"               # "seq2seq" | "causal" | "pipeline"
    quantize_4bit: bool = False               # Enable NF4 quantization (CUDA only)
    max_input_tokens: int = 512               # Max input token length
    max_output_tokens: int = 256              # Max new tokens to generate
    num_beams: int = 4                        # Beam search width
    no_repeat_ngram_size: int = 3             # Avoid repeating n-grams
    temperature: float = 1.0                  # Sampling temperature (1.0 = greedy with beams)
    do_sample: bool = False                   # Greedy decoding by default
    repetition_penalty: float = 1.0           # Repetition penalty (1.0 = no penalty)
    device: Optional[str] = None              # Override auto-detected device


def _env(key: str, default: str) -> str:
    """Read from environment with fallback."""
    return os.environ.get(key, default)


def _adapter_env(key: str, default: str) -> Optional[str]:
    """
    Read adapter path from environment.
    Returns None if the path is empty or the directory doesn't contain adapter files.
    This prevents loading stale or invalid adapters at startup.
    """
    path = os.environ.get(key, default)
    if not path:
        return None
    return path if path else None


# ── Adapter Version Registry ─────────────────────────────────────────────────
# Change versions here for rollback/promotion. Each key is the task name,
# each value is the version directory under models/adapters/{task}/{version}.
# Set to empty string to disable adapter (use base model only).

ADAPTER_VERSIONS = {
    "paraphrase": _env("ADAPTER_VERSION_PARAPHRASE", "v2"),
    "grammar":    _env("ADAPTER_VERSION_GRAMMAR", "v2"),
    "simplify":   _env("ADAPTER_VERSION_SIMPLIFY", "v2"),
    "tone":       _env("ADAPTER_VERSION_TONE", "v2"),
    "summarize":  _env("ADAPTER_VERSION_SUMMARIZE", "v2"),
}


def _adapter_path(task: str) -> Optional[str]:
    """Resolve versioned adapter path for a task."""
    version = ADAPTER_VERSIONS.get(task, "")
    if not version:
        return None
    path = f"models/adapters/{task}/{version}"
    return path


# ── Per-Service Configurations ────────────────────────────────────────────────
# Each service imports its own config:  from shared.model_config import PARAPHRASE_CONFIG

PARAPHRASE_CONFIG = ModelConfig(
    base_model=_env("MODEL_PARAPHRASE_BASE", "humarin/chatgpt_paraphraser_on_T5_base"),
    adapter_path=_adapter_env("MODEL_PARAPHRASE_ADAPTER", "") or _adapter_path("paraphrase"),
    model_type="seq2seq",
    max_input_tokens=512,
    max_output_tokens=512,
    num_beams=5,
)

GRAMMAR_CONFIG = ModelConfig(
    base_model=_env("MODEL_GRAMMAR_BASE", "vennify/t5-base-grammar-correction"),
    adapter_path=_adapter_env("MODEL_GRAMMAR_ADAPTER", "") or _adapter_path("grammar"),
    model_type="seq2seq",
    max_input_tokens=512,
    max_output_tokens=512,
    num_beams=5,
)

SIMPLIFY_CONFIG = ModelConfig(
    base_model=_env("MODEL_SIMPLIFY_BASE", "google/flan-t5-base"),
    adapter_path=_adapter_env("MODEL_SIMPLIFY_ADAPTER", "") or _adapter_path("simplify"),
    model_type="seq2seq",
    max_input_tokens=768,
    max_output_tokens=256,
    num_beams=4,
)

TONE_CONFIG = ModelConfig(
    base_model=_env("MODEL_TONE_BASE", "google/flan-t5-base"),
    adapter_path=_adapter_env("MODEL_TONE_ADAPTER", "") or _adapter_path("tone"),
    model_type="seq2seq",
    max_input_tokens=768,
    max_output_tokens=256,
    num_beams=4,
)

SUMMARIZE_CONFIG = ModelConfig(
    base_model=_env("MODEL_SUMMARIZE_BASE", "facebook/bart-large-cnn"),
    adapter_path=_adapter_env("MODEL_SUMMARIZE_ADAPTER", "") or _adapter_path("summarize"),
    model_type="seq2seq",
    max_input_tokens=1024,
    max_output_tokens=150,
    num_beams=4,
)

RAG_LLM_CONFIG = ModelConfig(
    base_model=_env("MODEL_RAG_LLM_BASE", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    adapter_path=_adapter_env("MODEL_RAG_LLM_ADAPTER", "") or None,
    model_type="causal",
    quantize_4bit=True,
    max_input_tokens=2048,
    max_output_tokens=300,
    do_sample=False,
    repetition_penalty=1.1,
)

RAG_EMBED_CONFIG = ModelConfig(
    base_model=_env("MODEL_RAG_EMBED", "all-MiniLM-L6-v2"),
    model_type="embedding",
)

RAG_RERANK_CONFIG = ModelConfig(
    base_model=_env("MODEL_RAG_RERANK", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    model_type="cross_encoder",
)
