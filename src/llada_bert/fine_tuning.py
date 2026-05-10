from __future__ import annotations

import math
import random

import torch
from transformers import DataCollatorWithPadding


class DiffusionMaskingCollator:
    def __init__(
        self,
        tokenizer,
        min_mask_ratio: float = 0.15,
        max_mask_ratio: float = 0.7,
    ) -> None:
        self.tokenizer = tokenizer
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.pad_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.pad_collator(examples)
        input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]
        labels = torch.full_like(input_ids, fill_value=-100)

        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
                for row in input_ids
            ],
            dtype=torch.bool,
        )
        valid_token_mask = (attention_mask == 1) & (~special_tokens_mask)

        for row_index in range(input_ids.size(0)):
            candidate_positions = torch.where(valid_token_mask[row_index])[0]
            if len(candidate_positions) == 0:
                continue

            timestep_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
            num_to_mask = max(1, int(math.ceil(timestep_ratio * len(candidate_positions))))
            selected_order = torch.randperm(len(candidate_positions))[:num_to_mask]
            mask_positions = candidate_positions[selected_order]

            labels[row_index, mask_positions] = input_ids[row_index, mask_positions]
            input_ids[row_index, mask_positions] = self.tokenizer.mask_token_id

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch
