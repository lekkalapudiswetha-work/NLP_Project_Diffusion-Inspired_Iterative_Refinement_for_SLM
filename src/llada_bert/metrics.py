from __future__ import annotations

from collections import Counter


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def exact_recovery(prediction: str, reference: str) -> float:
    return float(prediction.strip() == reference.strip())


def bleu_like(prediction: str, reference: str, max_n: int = 2) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum((pred_ngrams & ref_ngrams).values())
        precisions.append(overlap / max(1, sum(pred_ngrams.values())))
    brevity_penalty = min(1.0, len(pred_tokens) / max(1, len(ref_tokens)))
    return brevity_penalty * sum(precisions) / len(precisions)


def rouge1_like(prediction: str, reference: str) -> float:
    pred_counts = Counter(_tokenize(prediction))
    ref_counts = Counter(_tokenize(reference))
    overlap = sum((pred_counts & ref_counts).values())
    total = max(1, sum(ref_counts.values()))
    return overlap / total


def lcs_length(pred_tokens: list[str], ref_tokens: list[str]) -> int:
    rows = len(pred_tokens) + 1
    cols = len(ref_tokens) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_like(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    return lcs_length(pred_tokens, ref_tokens) / len(ref_tokens)


def aggregate_text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {}

    totals = {
        "bleu": 0.0,
        "rouge1": 0.0,
        "rougeL": 0.0,
        "exact_recovery": 0.0,
    }
    for prediction, reference in zip(predictions, references):
        totals["bleu"] += bleu_like(prediction, reference)
        totals["rouge1"] += rouge1_like(prediction, reference)
        totals["rougeL"] += rouge_l_like(prediction, reference)
        totals["exact_recovery"] += exact_recovery(prediction, reference)
    return {key: value / len(predictions) for key, value in totals.items()}
