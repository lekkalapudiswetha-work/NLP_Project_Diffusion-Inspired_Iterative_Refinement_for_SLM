from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass(slots=True)
class MaskedLMOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    confidences: torch.Tensor


class BertMaskedLMWrapper:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.vocab_size = self.model.config.vocab_size

    @property
    def special_token_ids(self) -> set[int]:
        return {
            token_id
            for token_id in (
                self.mask_token_id,
                self.pad_token_id,
                self.cls_token_id,
                self.sep_token_id,
            )
            if token_id is not None
        }

    def encode_prompt(self, prompt: str, max_length: int) -> torch.Tensor:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        return encoded["input_ids"].to(self.device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list[str]:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def build_generation_batch(
        self,
        prompt: str,
        sequence_length: int,
        batch_size: int,
    ) -> torch.Tensor:
        prompt_ids = self.encode_prompt(prompt, max_length=sequence_length)
        prompt_length = prompt_ids.shape[1]
        if prompt_length > sequence_length:
            raise ValueError("Prompt is longer than sequence length after tokenization.")

        batch = torch.full(
            (batch_size, sequence_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        batch[:, :prompt_length] = prompt_ids
        if prompt_length < sequence_length:
            batch[:, prompt_length:] = self.mask_token_id
        return batch

    @torch.inference_mode()
    def predict(self, input_ids: torch.Tensor, temperature: float = 1.0) -> MaskedLMOutput:
        logits = self.model(input_ids=input_ids).logits
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-5)
        probabilities = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(probabilities, dim=-1)
        return MaskedLMOutput(
            logits=logits,
            probabilities=probabilities,
            predictions=predictions,
            confidences=confidences,
        )

    @torch.inference_mode()
    def sample_predictions(
        self,
        probabilities: torch.Tensor,
        sample_mask: torch.Tensor,
        top_k: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        filtered = probabilities
        if top_k > 0:
            top_values, top_indices = torch.topk(probabilities, k=top_k, dim=-1)
            filtered = torch.zeros_like(probabilities).scatter(-1, top_indices, top_values)
            filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_probs = filtered[sample_mask]
        sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
        sampled_confidence = flat_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        return sampled, sampled_confidence

    def special_tokens_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            mask |= token_ids == token_id
        return mask

    def token_strings(self, token_ids: Iterable[int]) -> list[str]:
        return self.tokenizer.convert_ids_to_tokens(list(token_ids))
