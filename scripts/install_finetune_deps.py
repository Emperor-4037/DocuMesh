#!/usr/bin/env python3
"""
Fine-tuning dependencies installer.
Run before finetune.py to ensure all required packages are present.
"""
import subprocess, sys

PACKAGES = [
    "torch",
    "transformers>=4.40",
    "datasets",
    "peft>=0.10",
    "accelerate>=0.30",
    "bitsandbytes",
    "huggingface_hub",
    "sentencepiece",
    "protobuf",
]

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

if __name__ == "__main__":
    print("Installing fine-tuning dependencies...")
    for pkg in PACKAGES:
        print(f"  → {pkg}")
        install(pkg)
    print("Done. You can now run: python scripts/finetune.py --service all")
