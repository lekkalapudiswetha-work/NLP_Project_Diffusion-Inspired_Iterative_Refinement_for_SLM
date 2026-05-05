from __future__ import annotations

from dataclasses import dataclass

import torch

from .mlm_wrapper import BertMaskedLMWrapper
from .scheduler import DiffusionNoiseScheduler


@dataclass(slots=True)
class TaskBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    protected_mask: torch.Tensor
    task_type: str
    prompt_texts: list[str]
    reference_texts: list[str]


class TaskBuilder:
    def __init__(self, model: BertMaskedLMWrapper, scheduler: DiffusionNoiseScheduler) -> None:
        self.model = model
        self.scheduler = scheduler

    def _encode_texts(self, texts: list[str], sequence_length: int) -> torch.Tensor:
        rows = []
        for text in texts:
            encoded = self.model.encode_prompt(text, max_length=sequence_length).squeeze(0)
            if encoded.shape[0] < sequence_length:
                padding = torch.full(
                    (sequence_length - encoded.shape[0],),
                    fill_value=self.model.pad_token_id,
                    dtype=torch.long,
                    device=self.model.device,
                )
                encoded = torch.cat((encoded, padding), dim=0)
            rows.append(encoded[:sequence_length])
        return torch.stack(rows, dim=0)

    def build(
        self,
        texts: list[str],
        task_type: str,
        sequence_length: int,
        corruption_rate: float = 0.3,
        schedule_mask_ratio: float | None = None,
    ) -> TaskBatch:
        target_ids = self._encode_texts(texts, sequence_length)
        candidate_mask = (
            (target_ids != self.model.pad_token_id)
            & (target_ids != self.model.cls_token_id)
            & (target_ids != self.model.sep_token_id)
        )
        input_ids = target_ids.clone()

        if task_type == "infilling":
            mask_ratio = schedule_mask_ratio if schedule_mask_ratio is not None else corruption_rate
            input_ids = self.scheduler.mask_tokens(
                input_ids,
                candidate_mask=candidate_mask,
                mask_token_id=self.model.mask_token_id,
                mask_ratio=mask_ratio,
            )
            protected_mask = input_ids != self.model.mask_token_id
        elif task_type == "denoising":
            mask_ratio = schedule_mask_ratio if schedule_mask_ratio is not None else corruption_rate
            input_ids = self.scheduler.mask_tokens(
                input_ids,
                candidate_mask=candidate_mask,
                mask_token_id=self.model.mask_token_id,
                mask_ratio=mask_ratio,
            )
            protected_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            protected_mask |= input_ids == self.model.cls_token_id
            protected_mask |= input_ids == self.model.sep_token_id
            protected_mask |= input_ids == self.model.pad_token_id
        elif task_type == "iterative_generation":
            prompt_texts = []
            prompt_ids = target_ids.clone()
            for batch_index in range(prompt_ids.shape[0]):
                valid_tokens = torch.where(candidate_mask[batch_index])[0]
                keep_tokens = max(1, int(len(valid_tokens) * 0.1))
                keep_positions = set(valid_tokens[:keep_tokens].tolist())
                for token_index in valid_tokens.tolist():
                    if token_index not in keep_positions:
                        prompt_ids[batch_index, token_index] = self.model.mask_token_id
                prompt_texts.append(
                    self.model.decode(prompt_ids[batch_index], skip_special_tokens=True)[0]
                )
            input_ids = prompt_ids
            protected_mask = input_ids != self.model.mask_token_id
            return TaskBatch(
                input_ids=input_ids,
                target_ids=target_ids,
                protected_mask=protected_mask,
                task_type=task_type,
                prompt_texts=prompt_texts,
                reference_texts=texts,
            )
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        prompt_texts = self.model.decode(input_ids)
        return TaskBatch(
            input_ids=input_ids,
            target_ids=target_ids,
            protected_mask=protected_mask,
            task_type=task_type,
            prompt_texts=prompt_texts,
            reference_texts=texts,
        )
