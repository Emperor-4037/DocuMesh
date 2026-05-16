"""
Unified QLoRA fine-tuning pipeline for all AI Writing Assistant services.

Usage:
  python scripts/finetune.py --task paraphrase --epochs 3 --output models/adapters/paraphrase/v2
  python scripts/finetune.py --task all --dry-run

Design:
  - QLoRA (4-bit base + LoRA adapters) for maximum GPU efficiency
  - Per-architecture precision control: bf16 preferred, fp32 fallback for T5 (never fp16 on T5)
  - Gradient clipping + warmup + NaN detection for training stability
  - Conservative learning rates per backbone family
  - Produces adapter-only outputs (~10-50MB each)
  - Adapter versioned by output directory name
  - Rollback = point config at previous adapter directory

Changes from v1:
  - Fixed: T5 NaN loss by disabling fp16 for T5 architectures
  - Fixed: Simplify dataset loader via data_preprocessing module
  - Fixed: Grammar now uses both jfleg splits + expanded data
  - Fixed: Tone expanded from 9 templates to ~70 unique templates
  - Added: Gradient clipping (max_grad_norm=1.0)
  - Added: NaN loss detection with early stopping
  - Added: Per-run metadata.json for versioning
  - Added: Warmup ratio instead of fixed warmup steps
  - Added: Expanded LoRA target modules for better capacity
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from transformers import TrainerCallback


# ── Task Configurations ──────────────────────────────────────────────────────
# Key design decisions documented inline.

TASKS = {
    "paraphrase": {
        "base_model": "humarin/chatgpt_paraphraser_on_T5_base",
        "arch": "t5",          # Architecture family, controls precision logic
        "max_input_len": 128,
        "max_target_len": 128,
        "batch_size": 16,
        "lr": 5e-5,            # Lowered from 2e-4 to prevent gradient explosion on T5
        "epochs": 3,
        # T5 names attention layers as "q", "k", "v", "o"
        "target_modules": ["q", "k", "v", "o"],
    },
    "grammar": {
        "base_model": "vennify/t5-base-grammar-correction",
        "arch": "t5",
        "max_input_len": 128,
        "max_target_len": 128,
        "batch_size": 16,
        "lr": 5e-5,            # Conservative for small dataset
        "epochs": 5,           # More epochs to compensate for small dataset
        "target_modules": ["q", "k", "v", "o"],
    },
    "simplify": {
        "base_model": "google/flan-t5-base",
        "arch": "t5",
        "max_input_len": 256,
        "max_target_len": 128,
        "batch_size": 8,
        "lr": 5e-5,
        "epochs": 3,
        "target_modules": ["q", "k", "v", "o"],
    },
    "tone": {
        "base_model": "google/flan-t5-base",
        "arch": "t5",
        "max_input_len": 256,
        "max_target_len": 128,
        "batch_size": 8,
        "lr": 3e-5,            # Extra-conservative: synthetic data + T5 = high risk
        "epochs": 4,
        "target_modules": ["q", "k", "v", "o"],
    },
    "summarize": {
        "base_model": "facebook/bart-large-cnn",
        "arch": "bart",        # BART uses different attention layer names
        "max_input_len": 512,
        "max_target_len": 128,
        "batch_size": 4,
        "lr": 1e-4,            # BART is more stable with fp16; can use higher LR
        "epochs": 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
    },
}


def get_precision_config(arch: str, device: str) -> dict:
    """
    Return precision settings appropriate for the architecture.

    WHY: T5 models produce NaN loss under fp16 due to activation overflow in
    the relative position bias computation. bf16 is safe; fp32 is the fallback.
    BART models are stable under fp16.
    """
    import torch

    if device != "cuda":
        return {"fp16": False, "bf16": False, "compute_dtype": torch.float32}

    has_bf16 = torch.cuda.is_bf16_supported()

    if arch == "t5":
        # T5 MUST NOT use fp16. Use bf16 if available, else fp32.
        if has_bf16:
            return {"fp16": False, "bf16": True, "compute_dtype": torch.bfloat16}
        else:
            return {"fp16": False, "bf16": False, "compute_dtype": torch.float32}
    else:
        # BART and other architectures: bf16 preferred, fp16 acceptable
        if has_bf16:
            return {"fp16": False, "bf16": True, "compute_dtype": torch.bfloat16}
        else:
            return {"fp16": True, "bf16": False, "compute_dtype": torch.float16}


def save_run_metadata(output_dir: Path, task_name: str, cfg: dict,
                      precision: dict, train_result, dataset_sizes: dict):
    """
    Save a metadata.json alongside the adapter for versioning and rollback.
    This is the source of truth for what produced this adapter.
    """
    metadata = {
        "task": task_name,
        "base_model": cfg["base_model"],
        "architecture": cfg["arch"],
        "training_config": {
            "learning_rate": cfg["lr"],
            "epochs": cfg["epochs"],
            "batch_size": cfg["batch_size"],
            "max_input_len": cfg["max_input_len"],
            "max_target_len": cfg["max_target_len"],
            "target_modules": cfg["target_modules"],
            "lora_r": 8,
            "lora_alpha": 32,
            "gradient_clipping": 1.0,
        },
        "precision": {
            "fp16": precision["fp16"],
            "bf16": precision["bf16"],
        },
        "dataset": dataset_sizes,
        "results": {
            "train_loss": train_result.training_loss if train_result else None,
            "train_runtime_sec": train_result.metrics.get("train_runtime") if train_result else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "2.0.0",
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to {meta_path}")


class NaNLossCallback(TrainerCallback):

    """
    Halt training immediately if NaN loss is detected.
    WHY: Continuing training after NaN corrupts all subsequent gradients
    and produces a useless adapter. Fail fast, fix the cause.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        import math
        if logs and "loss" in logs:
            loss_val = logs["loss"]
            if isinstance(loss_val, (int, float)) and (math.isnan(loss_val) or math.isinf(loss_val)):
                print(f"\n  ⚠️  NaN/Inf loss detected at step {state.global_step}. Halting training.")
                control.should_training_stop = True


def train_task(task_name: str, output_dir: str, dry_run: bool = False, max_samples: int = 5000):
    """Run QLoRA fine-tuning for a single task with all stability fixes."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
        DataCollatorForSeq2Seq, BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    # Import our preprocessing module (lives in same directory)
    sys.path.insert(0, str(Path(__file__).parent))
    from data_preprocessing import load_and_preprocess

    cfg = TASKS[task_name]
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required but not available.")
    device = "cuda"

    precision = get_precision_config(cfg["arch"], device)

    print(f"\n{'='*60}")
    print(f"  Task       : {task_name}")
    print(f"  Model      : {cfg['base_model']}")
    print(f"  Arch       : {cfg['arch']}")
    print(f"  Device     : {device}")
    print(f"  Precision  : fp16={precision['fp16']}, bf16={precision['bf16']}")
    print(f"  LR         : {cfg['lr']}")
    print(f"  LoRA targets: {cfg['target_modules']}")
    print(f"  Output     : {output_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN] — skipping training")
        return

    start = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model + tokenizer ────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=precision["compute_dtype"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], legacy=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=precision["compute_dtype"],
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # ── Attach LoRA adapters ──────────────────────────────────────────────
    # Expanded target modules (q,k,v,o) for better capacity vs (q,v) only
    lora = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=cfg["target_modules"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ── Load + tokenize data ──────────────────────────────────────────────
    # Uses the preprocessing module for clean, deduplicated, validated data
    splits = load_and_preprocess(task_name, max_samples=max_samples)
    dataset_sizes = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }
    print(f"  Dataset: train={dataset_sizes['train']}, val={dataset_sizes['val']}, test={dataset_sizes['test']}")

    max_in, max_out = cfg["max_input_len"], cfg["max_target_len"]

    def tokenize(batch):
        inputs = tokenizer(batch["input"], max_length=max_in, truncation=True, padding="max_length")
        labels = tokenizer(text_target=batch["output"], max_length=max_out, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_ds = splits["train"].map(tokenize, batched=True, remove_columns=["input", "output"])
    val_ds = splits["val"].map(tokenize, batched=True, remove_columns=["input", "output"])

    # ── Training arguments ────────────────────────────────────────────────
    args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=2,
        learning_rate=cfg["lr"],
        # Precision: architecture-aware (never fp16 for T5)
        fp16=precision["fp16"],
        bf16=precision["bf16"],
        # Stability: gradient clipping prevents explosion
        max_grad_norm=1.0,
        # Warmup: ratio-based adapts to dataset size automatically
        warmup_ratio=0.1,
        # Evaluation and checkpointing
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_steps=20,
        report_to="none",
        # Reproducibility
        seed=42,
        data_seed=42,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[NaNLossCallback()],
    )

    train_result = trainer.train()

    # ── Save adapter + metadata ───────────────────────────────────────────
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    save_run_metadata(out, task_name, cfg, precision, train_result, dataset_sizes)

    elapsed = (time.time() - start) / 60
    final_loss = train_result.training_loss
    print(f"\n  [OK] {task_name} done in {elapsed:.1f} min -> {out}")
    print(f"       Final train loss: {final_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for AI Writing Assistant v2")
    parser.add_argument("--task", choices=list(TASKS.keys()) + ["all"], default="all")
    parser.add_argument("--output", default="models/adapters")
    parser.add_argument("--version", default="v2", help="Version tag for output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = list(TASKS.keys()) if args.task == "all" else [args.task]

    total_start = time.time()
    results = {}

    for task in tasks:
        if args.epochs:
            TASKS[task]["epochs"] = args.epochs
        if args.lr:
            TASKS[task]["lr"] = args.lr

        out_dir = f"{args.output}/{task}/{args.version}"
        try:
            train_task(task, out_dir, dry_run=args.dry_run, max_samples=args.max_samples)
            results[task] = "OK"
        except Exception as e:
            print(f"  [ERROR] Task {task} failed: {e}")
            import traceback
            traceback.print_exc()
            results[task] = f"FAILED: {e}"

    total_min = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print("  Training Summary")
    print(f"{'='*60}")
    for task, status in results.items():
        print(f"  {task:15s} : {status}")
    print(f"\n  Total time: {total_min:.1f} minutes")


if __name__ == "__main__":
    main()
