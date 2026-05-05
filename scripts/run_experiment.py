from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llada_bert import AblationRunner, ExperimentConfig
from llada_bert.data import load_experiment_texts


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small BERT diffusion text experiment")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--dataset-config", type=str)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="HF model id or local checkpoint directory, e.g. artifacts/llada_bert_finetuned",
    )
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--task-type", type=str, default="iterative_generation")
    parser.add_argument("--schedule-type", type=str, default="linear")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--corruption-rate", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--remask-strategy", type=str, default="low_confidence")
    parser.add_argument("--disable-thresholding", action="store_true")
    parser.add_argument("--disable-remasking", action="store_true")
    parser.add_argument("--disable-progressive-masking", action="store_true")
    parser.add_argument("--ablate-thresholds", type=float, nargs="*")
    parser.add_argument("--ablate-steps", type=int, nargs="*")
    parser.add_argument("--ablate-corruption-rates", type=float, nargs="*")
    parser.add_argument("--ablate-task-types", type=str, nargs="*")
    parser.add_argument("--limit", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.prompt and not args.input_file and not args.dataset_name:
        raise SystemExit("Provide one of --prompt, --input-file, or --dataset-name.")

    config = ExperimentConfig(
        model_name=args.model_name,
        device=args.device,
        task_type=args.task_type,
        schedule_type=args.schedule_type,
        steps=args.steps,
        sequence_length=args.sequence_length,
        temperature=args.temperature,
        threshold=args.threshold,
        corruption_rate=args.corruption_rate,
        top_k=args.top_k,
        seed=args.seed,
        num_samples=args.num_samples,
        remask_strategy=args.remask_strategy,
        enable_thresholding=not args.disable_thresholding,
        enable_remasking=not args.disable_remasking,
        enable_progressive_masking=not args.disable_progressive_masking,
    )
    runner = AblationRunner(config)
    texts = load_experiment_texts(
        prompt=args.prompt,
        input_file=args.input_file,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.dataset_split,
        text_column=args.text_column,
        limit=args.limit,
    )

    grid = {}
    if args.ablate_thresholds:
        grid["threshold"] = args.ablate_thresholds
    if args.ablate_steps:
        grid["steps"] = args.ablate_steps
    if args.ablate_corruption_rates:
        grid["corruption_rate"] = args.ablate_corruption_rates
    if args.ablate_task_types:
        grid["task_type"] = args.ablate_task_types

    if grid:
        output = runner.run_task_ablation_grid(texts, grid)
    else:
        output = runner.run_task_batch(texts)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
