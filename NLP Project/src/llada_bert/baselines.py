from __future__ import annotations

from dataclasses import dataclass

import torch

from .mlm_wrapper import BertMaskedLMWrapper
from .tasks import TaskBatch


@dataclass(slots=True)
class BaselineResult:
    token_ids: torch.Tensor
    texts: list[str]
    mean_confidence: float


class OnePassBertBaseline:
    def __init__(self, model: BertMaskedLMWrapper, top_k: int = 25) -> None:
        self.model = model
        self.top_k = top_k

    def predict(self, batch: TaskBatch, temperature: float = 1.0) -> BaselineResult:
        output = self.model.predict(batch.input_ids, temperature=temperature)
        masked_positions = batch.input_ids == self.model.mask_token_id
        current_ids = batch.input_ids.clone()
        confidences = output.confidences.clone()

        if masked_positions.any():
            sampled_tokens, sampled_confidences = self.model.sample_predictions(
                output.probabilities,
                masked_positions,
                top_k=self.top_k,
            )
            current_ids[masked_positions] = sampled_tokens
            confidences[masked_positions] = sampled_confidences

        return BaselineResult(
            token_ids=current_ids,
            texts=self.model.decode(current_ids),
            mean_confidence=float(confidences.mean().item()),
        )
