from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class NoiseStep:
    step: int
    total_steps: int
    noise_level: float
    threshold: float


class DiffusionNoiseScheduler:
    def __init__(
        self,
        total_steps: int,
        base_threshold: float = 0.85,
        threshold_annealing: float = 0.25,
        schedule_type: str = "linear",
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps
        self.base_threshold = base_threshold
        self.threshold_annealing = threshold_annealing
        self.schedule_type = schedule_type

    def noise_level(self, step: int) -> float:
        progress = step / max(self.total_steps - 1, 1)
        if self.schedule_type == "progressive":
            progressive_levels = (0.50, 0.30, 0.15, 0.05, 0.0)
            index = min(
                int(progress * (len(progressive_levels) - 1)),
                len(progressive_levels) - 1,
            )
            return progressive_levels[index]
        return max(0.0, 1.0 - progress)

    def threshold(self, step: int) -> float:
        progress = step / max(self.total_steps - 1, 1)
        return min(0.999, self.base_threshold - self.threshold_annealing * (1.0 - progress))

    def state(self, step: int) -> NoiseStep:
        return NoiseStep(
            step=step,
            total_steps=self.total_steps,
            noise_level=self.noise_level(step),
            threshold=self.threshold(step),
        )

    def mask_tokens(
        self,
        sequence: torch.Tensor,
        candidate_mask: torch.Tensor,
        mask_token_id: int,
        mask_ratio: float,
    ) -> torch.Tensor:
        updated = sequence.clone()
        for batch_index in range(sequence.shape[0]):
            positions = torch.where(candidate_mask[batch_index])[0]
            if len(positions) == 0:
                continue
            num_to_mask = min(len(positions), max(1, int(len(positions) * mask_ratio)))
            selected = positions[torch.randperm(len(positions), device=sequence.device)[:num_to_mask]]
            updated[batch_index, selected] = mask_token_id
        return updated

    def remask_low_confidence(
        self,
        input_ids: torch.Tensor,
        confidences: torch.Tensor,
        protected_mask: torch.Tensor,
        mask_token_id: int,
        step: int,
    ) -> torch.Tensor:
        state = self.state(step)
        remask_budget = int((~protected_mask).sum(dim=1).max().item() * state.noise_level)
        if remask_budget <= 0:
            return input_ids

        updated = input_ids.clone()
        candidate_conf = confidences.masked_fill(protected_mask, float("inf"))
        for batch_index in range(updated.shape[0]):
            batch_budget = min(remask_budget, (~protected_mask[batch_index]).sum().item())
            if batch_budget <= 0:
                continue
            selected = torch.topk(
                candidate_conf[batch_index],
                k=batch_budget,
                largest=False,
            ).indices
            updated[batch_index, selected] = mask_token_id
        return updated
