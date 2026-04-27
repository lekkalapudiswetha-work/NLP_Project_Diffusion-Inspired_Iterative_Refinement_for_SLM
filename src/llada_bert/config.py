from dataclasses import dataclass


@dataclass(slots=True)
class ExperimentConfig:
    model_name: str = "bert-base-uncased"
    device: str = "cpu"
    max_length: int = 32
    steps: int = 12
    sequence_length: int = 24
    temperature: float = 1.0
    threshold: float = 0.85
    remask_strategy: str = "low_confidence"
    top_k: int = 25
    seed: int = 0
    num_samples: int = 2
