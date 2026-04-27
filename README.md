# LLaDA-Inspired Text Diffusion with BERT

This project is a small, modular prototype for masked-token diffusion text generation using a pretrained BERT masked language model.

## Modules

1. `BertMaskedLMWrapper`: thin Hugging Face wrapper for masked-token scoring.
2. `DiffusionNoiseScheduler`: schedules corruption and target remasking.
3. `IterativeDenoisingSampler`: confidence-thresholded denoising loop.
4. `ConvergenceLogger`: tracks token churn and confidence over time.
5. `run_experiment.py`: simple ablation runner for laptop-scale experiments.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
python scripts/run_experiment.py \
  --prompt "The history of language models" \
  --steps 12 \
  --sequence-length 24 \
  --threshold 0.85 \
  --num-samples 2
```

## Notes

- The implementation uses `bert-base-uncased` by default.
- Generation is done by iterative unmasking and optional remasking, inspired by diffusion-style refinement rather than autoregression.
- Defaults are chosen for CPU or modest laptop GPU runs.
