from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class StepMetrics:
    step: int
    masked_tokens: int
    changed_tokens: int
    mean_confidence: float
    accepted_tokens: int


@dataclass
class ConvergenceLogger:
    history: list[StepMetrics] = field(default_factory=list)

    def log(
        self,
        step: int,
        previous_ids: torch.Tensor,
        current_ids: torch.Tensor,
        confidences: torch.Tensor,
        accepted_mask: torch.Tensor,
        mask_token_id: int,
    ) -> None:
        masked_tokens = int((current_ids == mask_token_id).sum().item())
        changed_tokens = int((previous_ids != current_ids).sum().item())
        mean_confidence = float(confidences.mean().item())
        accepted_tokens = int(accepted_mask.sum().item())
        self.history.append(
            StepMetrics(
                step=step,
                masked_tokens=masked_tokens,
                changed_tokens=changed_tokens,
                mean_confidence=mean_confidence,
                accepted_tokens=accepted_tokens,
            )
        )

    def summary(self) -> dict[str, float]:
        if not self.history:
            return {}
        return {
            "steps": float(len(self.history)),
            "final_masked_tokens": float(self.history[-1].masked_tokens),
            "final_changed_tokens": float(self.history[-1].changed_tokens),
            "final_mean_confidence": float(self.history[-1].mean_confidence),
        }

    def as_rows(self) -> list[dict[str, float | int]]:
        return [
            {
                "step": item.step,
                "masked_tokens": item.masked_tokens,
                "changed_tokens": item.changed_tokens,
                "mean_confidence": round(item.mean_confidence, 4),
                "accepted_tokens": item.accepted_tokens,
            }
            for item in self.history
        ]
