#!/usr/bin/env python3
"""
Download all models required by the AI Writing Assistant platform.
Run this once before starting docker-compose to pre-cache models locally.

Usage:
  python scripts/download_models.py           # download all
  python scripts/download_models.py --check   # just verify which are cached
"""
import argparse
import sys
from pathlib import Path

MODELS = [
    # (model_id, task_description)
    ("humarin/chatgpt_paraphraser_on_T5_base", "Paraphrase — T5-base fine-tuned for rewriting"),
    ("vennify/t5-base-grammar-correction",     "Grammar — T5-base fine-tuned on C4 grammar pairs"),
    ("google/flan-t5-base",                    "Simplify + Tone — Flan-T5 instruction-following"),
    ("facebook/bart-large-cnn",                "Summarize — BART-Large fine-tuned on CNN/DailyMail"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",    "RAG LLM — 1.1B chat model (4-bit on GPU)"),
    ("sentence-transformers/all-MiniLM-L6-v2", "RAG Embeddings — 22M sentence encoder"),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2",  "RAG Reranker — Cross-encoder for precision"),
]


def check_cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    result = try_to_load_from_cache(model_id, "config.json")
    return result is not None and result != "..."


def download_model(model_id: str, description: str):
    from transformers import AutoTokenizer, AutoConfig
    from huggingface_hub import snapshot_download

    print(f"\n  📥 {model_id}")
    print(f"     {description}")

    if check_cached(model_id):
        print(f"     ✅ Already cached — skipping")
        return

    try:
        snapshot_download(model_id, ignore_patterns=["*.msgpack", "flax_model*", "rust_model*"])
        print(f"     ✅ Downloaded")
    except Exception as e:
        print(f"     ❌ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download all AI platform models")
    parser.add_argument("--check", action="store_true", help="Only check cache status, don't download")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub transformers")
        sys.exit(1)

    print("🤖 AI Writing Assistant — Model Setup")
    print(f"   Models to prepare: {len(MODELS)}")
    print()

    for model_id, description in MODELS:
        cached = check_cached(model_id)
        if args.check:
            status = "✅ cached" if cached else "❌ not cached"
            print(f"  {status} — {model_id}")
        else:
            download_model(model_id, description)

    if not args.check:
        print("\n🎉 All models ready. You can now run: docker-compose up --build")


if __name__ == "__main__":
    main()
