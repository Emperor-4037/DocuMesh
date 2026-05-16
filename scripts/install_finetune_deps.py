#!/usr/bin/env python3
"""
Fine-tuning and evaluation dependencies installer.
Run before finetune.py to ensure all required packages are present.

Usage:
  python scripts/install_finetune_deps.py           # install all
  python scripts/install_finetune_deps.py --eval     # install eval deps only
"""
import subprocess, sys, argparse

TRAIN_PACKAGES = [
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

EVAL_PACKAGES = [
    "rouge-score",
    "bert-score",
    "nltk",
]

def install(pkg):
    cmd = [sys.executable, "-m", "pip", "install", pkg, "-q"]
    if pkg == "torch":
        cmd.extend(["--index-url", "https://download.pytorch.org/whl/cu124"])
    subprocess.check_call(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Install evaluation deps only")
    args = parser.parse_args()

    packages = EVAL_PACKAGES if args.eval else TRAIN_PACKAGES + EVAL_PACKAGES

    print("Installing dependencies...")
    for pkg in packages:
        print(f"  -> {pkg}")
        install(pkg)
    print("Done. You can now run:")
    print("  python scripts/finetune.py --task all")
    print("  python scripts/evaluate.py --task all")
