from __future__ import annotations

from dataclasses import asdict
from itertools import product

import torch

from .config import ExperimentConfig
from .mlm_wrapper import BertMaskedLMWrapper
from .sampler import IterativeDenoisingSampler
from .scheduler import DiffusionNoiseScheduler


class AblationRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self.model = BertMaskedLMWrapper(config.model_name, config.device)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_single(self, prompt: str, overrides: dict | None = None) -> dict:
        cfg = asdict(self.config)
        if overrides:
            cfg.update(overrides)

        scheduler = DiffusionNoiseScheduler(
            total_steps=cfg["steps"],
            base_threshold=cfg["threshold"],
        )
        sampler = IterativeDenoisingSampler(
            model=self.model,
            scheduler=scheduler,
            threshold=cfg["threshold"],
            top_k=cfg["top_k"],
        )
        result = sampler.sample(
            prompt=prompt,
            batch_size=cfg["num_samples"],
            sequence_length=cfg["sequence_length"],
            steps=cfg["steps"],
            temperature=cfg["temperature"],
            remask_strategy=cfg["remask_strategy"],
        )
        return {
            "config": cfg,
            "texts": result.texts,
            "metrics": result.logger.as_rows(),
            "summary": result.logger.summary(),
        }

    def run_ablation_grid(self, prompt: str, grid: dict[str, list]) -> list[dict]:
        keys = list(grid.keys())
        results = []
        for values in product(*(grid[key] for key in keys)):
            overrides = dict(zip(keys, values))
            results.append(self.run_single(prompt, overrides=overrides))
        return results
