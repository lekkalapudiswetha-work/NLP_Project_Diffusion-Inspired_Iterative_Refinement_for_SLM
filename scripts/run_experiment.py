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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small BERT diffusion text experiment")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--remask-strategy", type=str, default="low_confidence")
    parser.add_argument("--ablate-thresholds", type=float, nargs="*")
    parser.add_argument("--ablate-steps", type=int, nargs="*")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        model_name=args.model_name,
        device=args.device,
        steps=args.steps,
        sequence_length=args.sequence_length,
        temperature=args.temperature,
        threshold=args.threshold,
        top_k=args.top_k,
        seed=args.seed,
        num_samples=args.num_samples,
        remask_strategy=args.remask_strategy,
    )
    runner = AblationRunner(config)

    grid = {}
    if args.ablate_thresholds:
        grid["threshold"] = args.ablate_thresholds
    if args.ablate_steps:
        grid["steps"] = args.ablate_steps

    if grid:
        output = runner.run_ablation_grid(args.prompt, grid)
    else:
        output = runner.run_single(args.prompt)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
