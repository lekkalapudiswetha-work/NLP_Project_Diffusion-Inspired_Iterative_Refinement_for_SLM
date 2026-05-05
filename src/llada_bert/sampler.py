from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .logging_utils import ConvergenceLogger
from .mlm_wrapper import BertMaskedLMWrapper
from .scheduler import DiffusionNoiseScheduler


@dataclass(slots=True)
class SampleResult:
    token_ids: torch.Tensor
    texts: list[str]
    logger: ConvergenceLogger


class IterativeDenoisingSampler:
    def __init__(
        self,
        model: BertMaskedLMWrapper,
        scheduler: DiffusionNoiseScheduler,
        threshold: float = 0.85,
        top_k: int = 25,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.threshold = threshold
        self.top_k = top_k

    def sample(
        self,
        prompt: str,
        batch_size: int,
        sequence_length: int,
        steps: int,
        temperature: float = 1.0,
        remask_strategy: str = "low_confidence",
        initial_ids: torch.Tensor | None = None,
        protected_mask: torch.Tensor | None = None,
        enable_thresholding: bool = True,
        enable_remasking: bool = True,
    ) -> SampleResult:
        if initial_ids is None:
            current_ids = self.model.build_generation_batch(prompt, sequence_length, batch_size)
        else:
            current_ids = initial_ids.clone().to(self.model.device)
        if protected_mask is None:
            protected_mask = current_ids != self.model.mask_token_id
        logger = ConvergenceLogger()

        for step in range(steps):
            step_started = time.perf_counter()
            previous_ids = current_ids.clone()
            output = self.model.predict(current_ids, temperature=temperature)
            dynamic_threshold = self.scheduler.threshold(step)

            masked_positions = current_ids == self.model.mask_token_id
            sampled_ids = current_ids.clone()
            sampled_confidences = output.confidences.clone()

            if masked_positions.any():
                sampled_tokens, sampled_token_conf = self.model.sample_predictions(
                    output.probabilities,
                    masked_positions,
                    top_k=self.top_k,
                )
                sampled_ids[masked_positions] = sampled_tokens
                sampled_confidences[masked_positions] = sampled_token_conf

            if enable_thresholding:
                accept_mask = masked_positions & (sampled_confidences >= dynamic_threshold)
            else:
                accept_mask = masked_positions
            fallback_mask = masked_positions & ~accept_mask
            current_ids = torch.where(accept_mask, sampled_ids, current_ids)
            step_latency_ms = (time.perf_counter() - step_started) * 1000.0

            if not fallback_mask.any():
                logger.log(
                    step=step,
                    previous_ids=previous_ids,
                    current_ids=current_ids,
                    confidences=sampled_confidences,
                    accepted_mask=accept_mask,
                    mask_token_id=self.model.mask_token_id,
                    step_latency_ms=step_latency_ms,
                )
                if not (current_ids == self.model.mask_token_id).any():
                    break
            else:
                if enable_remasking and remask_strategy == "low_confidence":
                    current_ids = self.scheduler.remask_low_confidence(
                        input_ids=torch.where(masked_positions, sampled_ids, current_ids),
                        confidences=sampled_confidences,
                        protected_mask=protected_mask,
                        mask_token_id=self.model.mask_token_id,
                        step=step,
                    )
                else:
                    current_ids = torch.where(fallback_mask, current_ids, sampled_ids)

                logger.log(
                    step=step,
                    previous_ids=previous_ids,
                    current_ids=current_ids,
                    confidences=sampled_confidences,
                    accepted_mask=accept_mask,
                    mask_token_id=self.model.mask_token_id,
                    step_latency_ms=step_latency_ms,
                )

            if not (current_ids == self.model.mask_token_id).any():
                break

        texts = self.model.decode(current_ids)
        return SampleResult(token_ids=current_ids, texts=texts, logger=logger)
