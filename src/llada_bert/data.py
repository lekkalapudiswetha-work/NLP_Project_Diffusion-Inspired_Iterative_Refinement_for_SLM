from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_DATASET_SPLIT = "validation"
DEFAULT_TEXT_COLUMN = "text"


DEFAULT_TEXTS = [
    "Language models can improve reasoning through iterative refinement.",
    "Masked denoising objectives help models recover corrupted text.",
    "Diffusion-style generation revises uncertain tokens across multiple steps.",
]


def _clean_texts(texts: list[str], limit: int | None = None) -> list[str]:
    cleaned = [text.strip() for text in texts if text and text.strip()]
    if limit is not None:
        cleaned = cleaned[:limit]
    return cleaned


def load_texts(path: str | None = None, limit: int | None = None) -> list[str]:
    if path is None:
        texts = DEFAULT_TEXTS
    else:
        text_path = Path(path)
        texts = text_path.read_text().splitlines()
    return _clean_texts(texts, limit=limit)


def load_hf_texts(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    limit: int | None = None,
) -> list[str]:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if text_column not in dataset.column_names:
        raise ValueError(
            f"text_column '{text_column}' not found in dataset columns: {dataset.column_names}"
        )
    return _clean_texts(dataset[text_column], limit=limit)


def load_experiment_texts(
    prompt: str | None = None,
    input_file: str | None = None,
    dataset_name: str | None = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_DATASET_SPLIT,
    text_column: str = DEFAULT_TEXT_COLUMN,
    limit: int | None = None,
) -> list[str]:
    if prompt:
        return [prompt]
    if input_file:
        return load_texts(input_file, limit=limit)
    if dataset_name:
        return load_hf_texts(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            text_column=text_column,
            limit=limit,
        )
    return load_texts(limit=limit)


def load_hf_training_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    split: str = "train",
    text_column: str = DEFAULT_TEXT_COLUMN,
    limit: int | None = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    if text_column not in dataset.column_names:
        raise ValueError(
            f"text_column '{text_column}' not found in dataset columns: {dataset.column_names}"
        )
    dataset = dataset.filter(lambda example: bool(example[text_column] and example[text_column].strip()))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def tokenize_text_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    text_column: str = "text",
) -> Dataset:
    if text_column not in dataset.column_names:
        raise ValueError(
            f"text_column '{text_column}' not found in dataset columns: {dataset.column_names}"
        )

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
