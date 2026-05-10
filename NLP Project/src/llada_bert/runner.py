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

    def _resolved_cfg(self, overrides: dict | None = None) -> dict:
        cfg = asdict(self.config)
        if overrides:
            cfg.update(overrides)
        if cfg["task_type"] == "denoising" and "enable_remasking" not in (overrides or {}):
            cfg["enable_remasking"] = False
        return cfg

    def _compact_summary(self, result: dict) -> dict:
        diffusion_summary = result["diffusion"]["summary"]
        baseline_summary = result["baseline"]["summary"]
        return {
            "task_type": result["task"]["task_type"],
            "steps": result["config"]["steps"],
            "threshold": result["config"]["threshold"],
            "corruption_rate": result["config"]["corruption_rate"],
            "enable_thresholding": result["config"]["enable_thresholding"],
            "enable_remasking": result["config"]["enable_remasking"],
            "diffusion_bleu": round(diffusion_summary.get("bleu", 0.0), 4),
            "baseline_bleu": round(baseline_summary.get("bleu", 0.0), 4),
            "bleu_delta": round(
                diffusion_summary.get("bleu", 0.0) - baseline_summary.get("bleu", 0.0),
                4,
            ),
            "diffusion_rouge1": round(diffusion_summary.get("rouge1", 0.0), 4),
            "baseline_rouge1": round(baseline_summary.get("rouge1", 0.0), 4),
            "rouge1_delta": round(
                diffusion_summary.get("rouge1", 0.0) - baseline_summary.get("rouge1", 0.0),
                4,
            ),
            "final_masked_tokens": diffusion_summary.get("final_masked_tokens", 0.0),
            "final_mean_confidence": round(diffusion_summary.get("final_mean_confidence", 0.0), 4),
        }

    def run_single(self, prompt: str, overrides: dict | None = None) -> dict:
        return self.run_task_batch([prompt], overrides=overrides)

    def run_task_batch(self, texts: list[str], overrides: dict | None = None) -> dict:
        cfg = self._resolved_cfg(overrides)

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

        result = {
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
        result["comparison"] = self._compact_summary(result)
        return result

    def run_ablation_grid(self, prompt: str, grid: dict[str, list]) -> list[dict]:
        return self.run_task_ablation_grid([prompt], grid)

    def run_task_ablation_grid(self, texts: list[str], grid: dict[str, list]) -> dict:
        keys = list(grid.keys())
        results = []
        for values in product(*(grid[key] for key in keys)):
            overrides = dict(zip(keys, values))
            results.append(self.run_task_batch(texts, overrides=overrides))
        return {
            "results": results,
            "comparison_table": [item["comparison"] for item in results],
        }
