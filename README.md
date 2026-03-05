# Conditioned Comment Prediction (CCP)

Operational implementation for the empirical evaluation presented in [*Towards Simulating Social Media Users with LLMs: Evaluating the Operational Validity of Conditioned Comment Prediction*](https://arxiv.org/abs/2602.22752).

## Overview

This repository provides the computational pipeline required to reproduce the Conditioned Comment Prediction (CCP) task. The framework benchmarks the operational validity of open-weight ~8B parameter models (specifically Llama 3.1, Qwen 3, and Ministral) in simulating user-specific response behaviors. 

The core experimental design systematically contrasts the predictive efficacy of distinct contextual representations—predicting a user's reply to a stimulus by conditioning on raw behavioral traces versus synthesized descriptive personas. This evaluation is conducted across three distinct linguistic environments: English, German, and Luxembourgish.

## Artifacts and Resources

- **Preprint:** Detailed methodology and empirical findings are available on [arXiv:2602.22752](https://arxiv.org/abs/2602.22752).
- **Datasets and Checkpoints:** All associated corpora, behavioral trace datasets, and model artifacts are hosted in the [Conditioned Comment Prediction Hugging Face Collection](https://huggingface.co/collections/nsschw/conditioned-comment-prediction).

## Setup and Execution

The implementation is structured as the `echo` Python package. It requires a modern Python runtime to support the underlying computational requirements.

**Requirements:**
- Python $\ge$ 3.13

**Installation:**
Execute the following command at the repository root to install the package:
```bash
pip install .
```

## Operational Workflow

Following installation, the core model training pipeline is exposed via a streamlined command-line interface. To initiate the primary training sequence, execute:

```bash
echo-train
```
