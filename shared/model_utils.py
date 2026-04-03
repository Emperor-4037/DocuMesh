"""
Shared GPU detection and model loading utilities.
All services import from here for consistent device handling.
"""
import logging
import torch

logger = logging.getLogger("model_utils")


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on available hardware."""
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {name} ({vram:.1f} GB VRAM). Using CUDA.")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Apple MPS detected. Using MPS.")
    else:
        device = "cpu"
        logger.warning("No GPU detected. Running on CPU — expect slower inference.")
    return device


def get_dtype(device: str) -> torch.dtype:
    """Return best float dtype for the device."""
    if device == "cuda":
        # bfloat16 is more stable for training; float16 is fine for inference
        return torch.float16
    return torch.float32


DEVICE = get_device()
DTYPE = get_dtype(DEVICE)
