# LLaDA-Inspired Text Diffusion with BERT

This project is a small, modular prototype for masked-token diffusion text generation using a pretrained BERT masked language model.

## Overview

The goal of this repo is to explore a laptop-friendly version of diffusion-style text refinement using a pretrained masked language model instead of an autoregressive decoder. Rather than generating tokens left-to-right, the sampler starts from masked positions and iteratively fills or re-masks tokens based on model confidence.

This is inspired by LLaDA-style iterative denoising, but intentionally simplified for fast experimentation, ablations, and code readability.

## What is included

1. `BertMaskedLMWrapper`: thin Hugging Face wrapper for masked-token scoring.
2. `DiffusionNoiseScheduler`: schedules corruption and target remasking.
3. `IterativeDenoisingSampler`: confidence-thresholded denoising loop.
4. `ConvergenceLogger`: tracks token churn and confidence over time.
5. `run_experiment.py`: simple ablation runner for laptop-scale experiments.

## Project structure

```text
.
|-- LICENSE
|-- README.md
|-- notebooks/
|   |-- evaluation_and_plots_colab.ipynb
|   |-- inference_and_ablations_colab.ipynb
|   `-- train_llada_bert_colab.ipynb
|-- pyproject.toml
|-- requirements.txt
|-- scripts/
|   `-- run_experiment.py
`-- src/
    `-- llada_bert/
        |-- __init__.py
        |-- config.py
        |-- logging_utils.py
        |-- mlm_wrapper.py
        |-- runner.py
        |-- sampler.py
        `-- scheduler.py
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
python3 scripts/run_experiment.py \
  --prompt "The history of language models" \
  --steps 12 \
  --sequence-length 24 \
  --threshold 0.85 \
  --num-samples 2
```

## Example ablation run

```bash
python3 scripts/run_experiment.py \
  --prompt "Neural language models can improve" \
  --ablate-thresholds 0.75 0.85 0.92 \
  --ablate-steps 8 12
```

The script prints JSON containing:

- generated samples
- per-step convergence metrics
- a compact run summary for each configuration

## Colab notebooks

- `notebooks/train_llada_bert_colab.ipynb` fine-tunes BERT with a diffusion-style masking objective and then runs iterative denoising generation.
- `notebooks/inference_and_ablations_colab.ipynb` focuses on loading a base or fine-tuned checkpoint for generation and small ablation sweeps.
- `notebooks/evaluation_and_plots_colab.ipynb` visualizes convergence logs, compares ablation settings, and plots confidence and masking trends.

## Design notes

- The implementation uses `bert-base-uncased` by default.
- Generation is done by iterative unmasking and optional remasking, inspired by diffusion-style refinement rather than autoregression.
- Defaults are chosen for CPU or modest laptop GPU runs.
- The current code focuses on fast prototyping rather than benchmark-quality text generation.

## Current limitations

- This uses a pretrained MLM directly, without diffusion-specific fine-tuning.
- Quality is strongly limited by BERT's masked-token training objective.
- The sampler is useful for experimentation and ablations, but it is not yet a full research reproduction.

## Next ideas

- add dataset-based reconstruction evaluation
- add notebook-based visualization for convergence traces
- support alternative masked LMs such as RoBERTa
- compare remask strategies and threshold schedules
