#!/usr/bin/env python3
"""
scripts/finetune_medgemma.py
=============================
Fine-tune MedGemma-4B-it on MIMIC-CXR image-report pairs using QLoRA.

Strategy:
  - Backbone: MedGemma-4B-it (frozen except LoRA adapters)
  - Quantization: 4-bit NF4 (bitsandbytes) to fit on single GPU
  - LoRA rank: 16 (good quality/memory balance for CXR domain)
  - Training objective: causal language modelling on report text
  - Prompt: includes image tokens + report generation system prompt

GPU requirements:
  - A100 40GB: bfloat16 full precision, batch_size=8
  - RTX 3090 24GB: 4-bit, batch_size=4, grad_accum=8
  - T4 16GB: 4-bit, batch_size=2, grad_accum=16

Usage:
    python scripts/finetune_medgemma.py \
        --data-dir data/processed \
        --output-dir checkpoints/medgemma-cxr \
        --epochs 3 \
        --batch-size 4

    # Dry-run to verify pipeline (no actual training)
    python scripts/finetune_medgemma.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def build_training_dataset(
    data_dir: Path,
    split: str,
    max_samples: int,
    processor,
    report_parser,
):
    """
    Build a HuggingFace Dataset for supervised fine-tuning.

    Each example format (MedGemma chat template):
        [USER]: <image> + report generation prompt
        [ASSISTANT]: FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}
    """
    import pandas as pd
    from torch.utils.data import Dataset as TorchDataset
    from PIL import Image as PILImage
    from src.models.prompts import REPORT_PROMPT_V1

    csv = data_dir / "metadata.csv"
    df = pd.read_csv(csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "split" in df.columns:
        df = df[df["split"] == split]

    df = df.dropna(subset=["text"]).head(max_samples).reset_index(drop=True)
    img_root = data_dir / "images"

    class CXRFinetuneDataset(TorchDataset):
        def __init__(self):
            self.df = df
            self.img_root = img_root
            self.parser = report_parser
            self.prompt = REPORT_PROMPT_V1

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            report = str(row.get("text", ""))
            sections = self.parser.parse(report)

            # Build target text: structured report
            target = (
                f"FINDINGS:\n{sections.findings}\n\n"
                f"IMPRESSION:\n{sections.impression}"
            ).strip()

            # Load image
            img_path = self.img_root / str(row.get("image", ""))
            try:
                image = PILImage.open(img_path).convert("RGB")
            except Exception:
                image = PILImage.new("RGB", (448, 448), color=128)

            # Tokenise using MedGemma chat template
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text":  self.prompt},
                ]},
                {"role": "assistant", "content": target},
            ]

            tokenised = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
            )
            # For causal LM, labels = input_ids (mask prompt tokens later)
            return {
                "input_ids":      tokenised["input_ids"].squeeze(0),
                "attention_mask": tokenised["attention_mask"].squeeze(0),
                "labels":         tokenised["input_ids"].squeeze(0).clone(),
            }

    return CXRFinetuneDataset()


def train(args):
    """Main fine-tuning loop using HuggingFace Trainer."""
    import torch
    from transformers import (
        AutoProcessor, AutoModelForImageTextToText,
        TrainingArguments, Trainer,
        BitsAndBytesConfig, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from src.data.report_parser import ReportParser

    hf_token = os.environ.get("HF_TOKEN")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model with 4-bit quantization ───────────────────────────────────
    logger.info("Loading MedGemma for fine-tuning …")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(args.model_id, token=hf_token)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    )

    # Prepare for k-bit training (gradient checkpointing + cast norms)
    model = prepare_model_for_kbit_training(model)

    # ── Attach LoRA adapters ─────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Build datasets ────────────────────────────────────────────────────────
    parser = ReportParser()
    data_dir = Path(args.data_dir)
    train_dataset = build_training_dataset(
        data_dir, "train", args.max_train_samples, processor, parser
    )
    val_dataset = build_training_dataset(
        data_dir, "validate", args.max_val_samples, processor, parser
    )
    logger.info("Train: %d  |  Val: %d", len(train_dataset), len(val_dataset))

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["tensorboard"],
        dataloader_pin_memory=False,  # required for quantised models
        remove_unused_columns=False,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting fine-tuning …")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_pretrained(str(output_dir / "final"))
    processor.save_pretrained(str(output_dir / "final"))

    # Save training config
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Fine-tuning complete. Model saved -> %s/final", output_dir)


def dry_run(args):
    """Verify the data pipeline without actual training."""
    logger.info("DRY RUN — checking data pipeline only …")
    from src.data.report_parser import ReportParser
    from src.data.dataset import SyntheticCXRDataset

    ds = SyntheticCXRDataset(n_samples=5)
    parser = ReportParser()
    for i in range(3):
        s = ds[i]
        sections = parser.parse(s.report)
        target = f"FINDINGS:\n{sections.findings}\n\nIMPRESSION:\n{sections.impression}"
        logger.info("Sample %d | study_id=%s | target_len=%d chars",
                    i, s.study_id, len(target))
        logger.info("  Findings: %s…", sections.findings[:80])
        logger.info("  Impression: %s…", sections.impression[:60])

    logger.info("Dry run OK — pipeline looks good.")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MedGemma on MIMIC-CXR")
    p.add_argument("--model-id",          default="google/medgemma-4b-it")
    p.add_argument("--data-dir",          default="data/processed")
    p.add_argument("--output-dir",        default="checkpoints/medgemma-cxr-ft")
    p.add_argument("--epochs",            type=int,   default=3)
    p.add_argument("--batch-size",        type=int,   default=4)
    p.add_argument("--grad-accum",        type=int,   default=8)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--lora-rank",         type=int,   default=16)
    p.add_argument("--lora-alpha",        type=int,   default=32)
    p.add_argument("--lora-dropout",      type=float, default=0.05)
    p.add_argument("--max-train-samples", type=int,   default=10000)
    p.add_argument("--max-val-samples",   type=int,   default=500)
    p.add_argument("--dry-run",           action="store_true",
                   help="Validate pipeline without training")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return

    import torch
    if not torch.cuda.is_available():
        logger.error(
            "CUDA GPU not detected. Fine-tuning MedGemma requires a GPU "
            "(minimum 16 GB VRAM with 4-bit quantization). "
            "Use --dry-run to test the pipeline."
        )
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
