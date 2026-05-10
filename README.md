# LLaDA-Inspired Text Diffusion with BERT

This project is a small, modular prototype for masked-token diffusion text generation using a pretrained BERT masked language model. The official default dataset is WikiText-2 raw via Hugging Face: `wikitext` / `wikitext-2-raw-v1`.

## Overview

The goal of this repo is to explore a laptop-friendly version of diffusion-style text refinement using a pretrained masked language model instead of an autoregressive decoder. Rather than generating tokens left-to-right, the sampler starts from masked positions and iteratively fills or re-masks tokens based on model confidence.

This is inspired by LLaDA-style iterative denoising, but intentionally simplified for fast experimentation, ablations, and code readability.

## What is included

1. `BertMaskedLMWrapper`: thin Hugging Face wrapper for masked-token scoring.
2. `DiffusionNoiseScheduler`: schedules corruption and target remasking.
3. `IterativeDenoisingSampler`: confidence-thresholded denoising loop.
4. `ConvergenceLogger`: tracks token churn and confidence over time.
5. `TaskBuilder` and baselines: infilling, denoising, and iterative generation tasks plus one-pass BERT baseline.
6. `run_experiment.py`: task-aware ablation runner for laptop-scale experiments.

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
|   |-- train_diffusion_mlm.py
|   `-- run_experiment.py
`-- src/
    `-- llada_bert/
        |-- __init__.py
        |-- baselines.py
        |-- config.py
        |-- data.py
        |-- fine_tuning.py
        |-- logging_utils.py
        |-- metrics.py
        |-- mlm_wrapper.py
        |-- runner.py
        |-- sampler.py
        |-- scheduler.py
        `-- tasks.py
```

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
python3 scripts/run_experiment.py \
  --task-type infilling \
  --limit 4 \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split validation \
  --text-column text \
  --steps 12 \
  --sequence-length 24 \
  --threshold 0.85 \
  --num-samples 2
```

## Training

Use the dedicated training script to fine-tune BERT with the diffusion-style masking objective outside the notebook:

```bash
python3 scripts/train_diffusion_mlm.py \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --train-split train \
  --eval-split validation \
  --text-column text \
  --train-limit 4000 \
  --eval-limit 500 \
  --output-dir artifacts/llada_bert_finetuned
```

## Quick Prompt Example

```bash
python3 scripts/run_experiment.py \
  --prompt "The history of language models" \
  --task-type infilling \
  --steps 12 \
  --sequence-length 24 \
  --threshold 0.85 \
  --num-samples 2
```

## Example ablation run

```bash
python3 scripts/run_experiment.py \
  --input-file sample_texts.txt \
  --task-type denoising \
  --ablate-thresholds 0.75 0.85 0.92 \
  --ablate-steps 1 3 5 10 \
  --ablate-corruption-rates 0.1 0.3 0.5
```

## Real datasets

You can now run experiments from a Hugging Face dataset instead of only inline prompts or local text files. If you do not specify a source explicitly, the repo defaults to WikiText-2 raw.

Example with WikiText-2:

```bash
python3 scripts/run_experiment.py \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --dataset-split validation \
  --text-column text \
  --task-type denoising \
  --limit 8
```

The data utilities also expose:

- `load_hf_texts(...)` for evaluation text loading
- `load_hf_training_dataset(...)` for training splits
- `tokenize_text_dataset(...)` for tokenization before fine-tuning

The script prints JSON containing:

- diffusion outputs and baseline outputs
- per-step convergence metrics
- latency, stability, and confidence summaries
- lightweight BLEU/ROUGE-style and exact-recovery metrics

## Colab notebooks

- `notebooks/train_llada_bert_colab.ipynb` fine-tunes BERT with a diffusion-style masking objective and then runs iterative denoising generation.
- `notebooks/inference_and_ablations_colab.ipynb` focuses on loading a base or fine-tuned checkpoint for generation and small ablation sweeps.
- `notebooks/evaluation_and_plots_colab.ipynb` visualizes convergence logs, compares ablation settings, and plots confidence and masking trends.

## Design notes

- The implementation uses `bert-base-uncased` by default.
- Generation is done by iterative unmasking and optional remasking, inspired by diffusion-style refinement rather than autoregression.
- Supported task modes are `infilling`, `denoising`, and `iterative_generation`.
- Supported ablation toggles include thresholding, remasking, and schedule choice.
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
