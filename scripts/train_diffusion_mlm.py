from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llada_bert import (  # noqa: E402
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_NAME,
    DEFAULT_TEXT_COLUMN,
    DiffusionMaskingCollator,
    load_hf_training_dataset,
    tokenize_text_dataset,
)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune BERT with a diffusion-style masking objective")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config", type=str, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--text-column", type=str, default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--train-limit", type=int, default=4000)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="artifacts/llada_bert_finetuned")
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--min-mask-ratio", type=float, default=0.15)
    parser.add_argument("--max-mask-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=default_device())
    return parser


def main() -> None:
    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(args.device)

    train_dataset = load_hf_training_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.train_split,
        text_column=args.text_column,
        limit=args.train_limit,
    )
    eval_dataset = load_hf_training_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.eval_split,
        text_column=args.text_column,
        limit=args.eval_limit,
    )

    train_dataset = tokenize_text_dataset(
        train_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_column=args.text_column,
    )
    eval_dataset = tokenize_text_dataset(
        eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        text_column=args.text_column,
    )

    collator = DiffusionMaskingCollator(
        tokenizer=tokenizer,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
        fp16=torch.cuda.is_available() and args.device.startswith("cuda"),
        dataloader_pin_memory=args.device.startswith("cuda"),
        remove_unused_columns=False,
        seed=args.seed,
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "output_dir": args.output_dir,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
