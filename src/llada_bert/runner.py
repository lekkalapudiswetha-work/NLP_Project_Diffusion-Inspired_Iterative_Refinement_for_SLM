from __future__ import annotations

from dataclasses import asdict
from itertools import product

import torch

from .baselines import OnePassBertBaseline
from .config import ExperimentConfig
from .metrics import aggregate_text_metrics
from .mlm_wrapper import BertMaskedLMWrapper
from .sampler import IterativeDenoisingSampler
from .scheduler import DiffusionNoiseScheduler
from .tasks import TaskBuilder


class AblationRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self.model = BertMaskedLMWrapper(config.model_name, config.device)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_scheduler(self, cfg: dict) -> DiffusionNoiseScheduler:
        return DiffusionNoiseScheduler(
            total_steps=cfg["steps"],
            base_threshold=cfg["threshold"],
            schedule_type=cfg["schedule_type"],
        )

    def run_single(self, prompt: str, overrides: dict | None = None) -> dict:
        return self.run_task_batch([prompt], overrides=overrides)

    def run_task_batch(self, texts: list[str], overrides: dict | None = None) -> dict:
        cfg = asdict(self.config)
        if overrides:
            cfg.update(overrides)

        scheduler = self._make_scheduler(cfg)
        task_builder = TaskBuilder(self.model, scheduler)
        task_batch = task_builder.build(
            texts=texts,
            task_type=cfg["task_type"],
            sequence_length=cfg["sequence_length"],
            corruption_rate=cfg["corruption_rate"],
            schedule_mask_ratio=cfg["corruption_rate"] if cfg["enable_progressive_masking"] else None,
        )

        sampler = IterativeDenoisingSampler(
            model=self.model,
            scheduler=scheduler,
            threshold=cfg["threshold"],
            top_k=cfg["top_k"],
        )
        diffusion_result = sampler.sample(
            prompt=task_batch.prompt_texts[0] if task_batch.prompt_texts else "",
            batch_size=len(texts),
            sequence_length=cfg["sequence_length"],
            steps=cfg["steps"],
            temperature=cfg["temperature"],
            remask_strategy=cfg["remask_strategy"],
            initial_ids=task_batch.input_ids,
            protected_mask=task_batch.protected_mask,
            enable_thresholding=cfg["enable_thresholding"],
            enable_remasking=cfg["enable_remasking"],
        )

        baseline = OnePassBertBaseline(self.model, top_k=cfg["top_k"])
        baseline_result = baseline.predict(task_batch, temperature=cfg["temperature"])

        diffusion_metrics = aggregate_text_metrics(diffusion_result.texts, task_batch.reference_texts)
        baseline_metrics = aggregate_text_metrics(baseline_result.texts, task_batch.reference_texts)

        return {
            "config": cfg,
            "task": {
                "task_type": task_batch.task_type,
                "inputs": task_batch.prompt_texts,
                "references": task_batch.reference_texts,
            },
            "diffusion": {
                "texts": diffusion_result.texts,
                "convergence": diffusion_result.logger.as_rows(),
                "summary": {
                    **diffusion_result.logger.summary(),
                    **diffusion_metrics,
                },
            },
            "baseline": {
                "name": "one_pass_bert",
                "texts": baseline_result.texts,
                "summary": {
                    "mean_confidence": baseline_result.mean_confidence,
                    **baseline_metrics,
                },
            },
        }

    def run_ablation_grid(self, prompt: str, grid: dict[str, list]) -> list[dict]:
        return self.run_task_ablation_grid([prompt], grid)

    def run_task_ablation_grid(self, texts: list[str], grid: dict[str, list]) -> list[dict]:
        keys = list(grid.keys())
        results = []
        for values in product(*(grid[key] for key in keys)):
            overrides = dict(zip(keys, values))
            results.append(self.run_task_batch(texts, overrides=overrides))
        return results
