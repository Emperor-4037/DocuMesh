#!/usr/bin/env python3
"""
Fine-tune all NLP services with LoRA/QLoRA adapters.

Strategy (to stay under 3 hours total on a single consumer GPU):
  - Use LoRA (r=8) adapters on T5/BART encoder-decoder models
  - Short training: 3 epochs on small high-quality dataset subsets
  - 4-bit quantization (QLoRA) if CUDA + bitsandbytes available
  - Save adapters to ./models/{service_name}/ for deployment

Models + Datasets:
  paraphrase  : humarin/chatgpt_paraphraser_on_T5_base  |  paws (PAWS-X paraphrase pairs)
  grammar     : vennify/t5-base-grammar-correction        |  jfleg (grammatical error correction)
  simplify    : google/flan-t5-base                       |  wiki_simple (Wikipedia/Simple Wikipedia aligned)
  tone        : google/flan-t5-base                       |  synthetic tone prompt-completion pairs (generated)
  summarize   : facebook/bart-large-cnn                   |  cnn_dailymail (already near-optimal, short finetune)

Usage:
  python scripts/finetune.py --service all           # Fine-tune all services
  python scripts/finetune.py --service paraphrase    # Fine-tune one
  python scripts/finetune.py --service all --dry-run # Preview config
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_QLORA = DEVICE == "cuda"

# Time budget per service (total 3h = 180 min -> ~30 min each for 5 services)
TIME_BUDGET_MINUTES = 30

SERVICE_CONFIGS = {
    "paraphrase": {
        "base_model": "humarin/chatgpt_paraphraser_on_T5_base",
        "dataset_name": "paws",
        "dataset_config": "labeled_final",
        "input_col": "sentence1",
        "target_col": "sentence2",
        "input_prefix": "paraphrase: ",
        "max_input_len": 128,
        "max_target_len": 128,
        "num_train_samples": 5000,
        "num_epochs": 3,
        "batch_size": 16,
        "lr": 2e-4,
    },
    "grammar": {
        "base_model": "vennify/t5-base-grammar-correction",
        "dataset_name": "jfleg",
        "dataset_config": None,
        "input_col": "sentence",
        "target_col": "corrections",  # list — handled specially
        "input_prefix": "grammar: ",
        "max_input_len": 128,
        "max_target_len": 128,
        "num_train_samples": 4000,
        "num_epochs": 3,
        "batch_size": 16,
        "lr": 2e-4,
    },
    "simplify": {
        "base_model": "google/flan-t5-base",
        "dataset_name": "wiki_lingua",
        "dataset_config": "english",
        "input_col": "source_paragraph",
        "target_col": "target_paragraph",
        "input_prefix": "Simplify the following text in plain English: ",
        "max_input_len": 256,
        "max_target_len": 128,
        "num_train_samples": 5000,
        "num_epochs": 3,
        "batch_size": 8,
        "lr": 2e-4,
    },
    "tone": {
        "base_model": "google/flan-t5-base",
        "dataset_name": "synthetic",  # generated inline
        "dataset_config": None,
        "input_col": "input",
        "target_col": "output",
        "input_prefix": "",  # prefix built into input
        "max_input_len": 256,
        "max_target_len": 128,
        "num_train_samples": 3000,
        "num_epochs": 4,
        "batch_size": 8,
        "lr": 2e-4,
    },
    "summarize": {
        "base_model": "facebook/bart-large-cnn",
        "dataset_name": "cnn_dailymail",
        "dataset_config": "3.0.0",
        "input_col": "article",
        "target_col": "highlights",
        "input_prefix": "",
        "max_input_len": 512,
        "max_target_len": 128,
        "num_train_samples": 3000,
        "num_epochs": 2,
        "batch_size": 4,
        "lr": 1e-4,
    },
}

# ── LoRA Configuration ─────────────────────────────────────────────────────────

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "SEQ_2_SEQ_LM",
    "target_modules": ["q", "v"],  # T5/BART attention projections
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ["transformers", "datasets", "peft", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if USE_QLORA:
        try:
            import bitsandbytes  # noqa
        except ImportError:
            missing.append("bitsandbytes")
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def build_bnb_config():
    if not USE_QLORA:
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def generate_tone_dataset(n: int = 3000):
    """Generate synthetic tone transfer pairs (formal <-> informal)."""
    from datasets import Dataset
    import random

    templates = {
        "formal": [
            ("hey, can you help me out?", "Could you please assist me with this matter?"),
            ("I wanna know what's up with the report", "I would like to inquire about the status of the report."),
            ("this is super important!!!", "This matter is of considerable importance."),
            ("let me know asap", "Please inform me at your earliest convenience."),
            ("the meeting's been pushed", "The meeting has been rescheduled."),
            ("thanks a bunch!", "I sincerely appreciate your assistance."),
            ("gotta fix this bug today", "It is necessary to resolve this issue today."),
            ("the deadline got moved up", "The deadline has been advanced."),
        ],
        "informal": [
            ("I would like to formally request your assistance.", "Hey, can you help me out?"),
            ("Please be advised that the meeting is rescheduled.", "FYI, the meeting's been moved."),
            ("Your cooperation in this matter is greatly appreciated.", "Really appreciate you helping!"),
        ],
    }

    data = {"input": [], "output": []}
    tones = list(templates.keys())
    for _ in range(n):
        tone = random.choice(tones)
        pair = random.choice(templates[tone])
        prefix = f"Rewrite in {tone} tone: {pair[0]}"
        data["input"].append(prefix)
        data["output"].append(pair[1])

    return Dataset.from_dict(data)


def load_dataset_for_service(config: dict):
    from datasets import load_dataset

    name = config["dataset_name"]
    cfg = config["dataset_config"]

    if name == "synthetic":
        return generate_tone_dataset(config["num_train_samples"])

    if name == "jfleg":
        ds = load_dataset("jfleg", split="validation")
        # jfleg corrections is a list; take first correction
        records = [
            {"input": s, "output": c[0] if c else s}
            for s, c in zip(ds["sentence"], ds["corrections"])
            if c
        ]
        from datasets import Dataset
        records = records[: config["num_train_samples"]]
        flat = {"input": [r["input"] for r in records], "output": [r["output"] for r in records]}
        return Dataset.from_dict(flat)

    print(f"  Loading dataset '{name}' (config={cfg})...")
    splits = load_dataset(name, cfg) if cfg else load_dataset(name)
    train = splits["train"]
    n = config["num_train_samples"]
    if len(train) > n:
        train = train.shuffle(seed=42).select(range(n))

    # Normalize to input/output columns
    in_col = config["input_col"]
    out_col = config["target_col"]

    if in_col not in train.column_names:
        # Handle nested columns
        sample = train[0]
        print(f"  Available columns: {list(sample.keys())}")
        raise ValueError(f"Column '{in_col}' not found in dataset.")

    prefix = config["input_prefix"]
    if name == "cnn_dailymail":
        return train.map(
            lambda x: {"input": x[in_col][:1024], "output": x[out_col]},
            remove_columns=train.column_names,
        )

    return train.map(
        lambda x: {"input": prefix + str(x[in_col])[:512], "output": str(x[out_col])[:256]},
        remove_columns=train.column_names,
    )


def finetune_service(service_name: str, dry_run: bool = False):
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    config = SERVICE_CONFIGS[service_name]
    output_dir = MODELS_DIR / service_name
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {service_name}")
    print(f"  Base model : {config['base_model']}")
    print(f"  Device     : {DEVICE} ({'QLoRA 4-bit' if USE_QLORA else 'float32'})")
    print(f"  Epochs     : {config['num_epochs']} | Batch: {config['batch_size']} | LR: {config['lr']}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] — skipping actual training")
        return

    start = time.time()

    # ── Load tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], legacy=False)

    # ── Load model with optional 4-bit quantization ──────────────────────────
    bnb = build_bnb_config()
    if bnb:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"]).to(DEVICE)

    # ── Attach LoRA adapters ─────────────────────────────────────────────────
    # Determine target modules based on model architecture
    if "bart" in config["base_model"].lower():
        targets = ["q_proj", "v_proj"]
    else:
        targets = ["q", "v"]

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=targets,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Load and tokenize dataset ────────────────────────────────────────────
    print(f"  Loading dataset...")
    dataset = load_dataset_for_service(config)

    max_in = config["max_input_len"]
    max_out = config["max_target_len"]

    def tokenize(batch):
        model_inputs = tokenizer(batch["input"], max_length=max_in, truncation=True, padding="max_length")
        labels = tokenizer(text_target=batch["output"], max_length=max_out, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("  Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # 10% validation split
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # ── Training arguments ───────────────────────────────────────────────────
    # Derive max steps from time budget
    steps_per_epoch = len(train_ds) // config["batch_size"]
    max_steps = min(
        steps_per_epoch * config["num_epochs"],
        (TIME_BUDGET_MINUTES * 60) // 4,  # rough: 4s/step
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        max_steps=max_steps,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=2,
        learning_rate=config["lr"],
        fp16=USE_QLORA,
        bf16=False,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=max(50, max_steps // 10),
        save_strategy="steps",
        save_steps=max(100, max_steps // 5),
        load_best_model_at_end=True,
        logging_steps=20,
        warmup_steps=min(100, max_steps // 10),
        report_to="none",
        dataloader_num_workers=2,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print(f"  Training for up to {max_steps} steps...")
    trainer.train()

    # ── Save LoRA adapter only (small, ~10-50MB vs full model) ───────────────
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    elapsed = (time.time() - start) / 60
    print(f"\n  ✅ {service_name} done in {elapsed:.1f} min → saved to {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune AI platform NLP services with LoRA/QLoRA")
    parser.add_argument(
        "--service",
        choices=list(SERVICE_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which service to fine-tune (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config without training")
    args = parser.parse_args()

    check_dependencies()

    total_start = time.time()
    services = list(SERVICE_CONFIGS.keys()) if args.service == "all" else [args.service]

    print(f"\n🚀 AI Platform Fine-tuning")
    print(f"   GPU    : {DEVICE.upper()}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    print(f"   Mode   : {'QLoRA 4-bit' if USE_QLORA else 'LoRA FP32'}")
    print(f"   Budget : {TIME_BUDGET_MINUTES} min/service")
    print(f"   Output : {MODELS_DIR}")

    for svc in services:
        finetune_service(svc, dry_run=args.dry_run)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n🎉 All services done in {total_elapsed:.1f} minutes total.")
    print(f"   Adapters saved to: {MODELS_DIR}")
    print(f"   Each service now loads from its adapter directory automatically.")


if __name__ == "__main__":
    main()
