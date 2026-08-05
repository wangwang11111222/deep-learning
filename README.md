# Deep Learning Systems

Deep-learning engineering projects covering context distillation for LLMs and sequence-model architecture design.

## Projects

### 1. Context Distillation

Implements a context-distillation workflow inspired by teacher/student reasoning transfer:

- Teacher model generates reasoning-style sentiment explanations.
- Heuristic answer extraction turns teacher output into labels.
- Student model is fine-tuned as a sequence classifier.
- Baseline compares against vanilla fine-tuning without teacher-generated reasoning.

Files:

- `final_project/contextdistillation5.py`: teacher generation, label extraction, and student training pipeline.
- `final_project/vanillaFineTune.py`: direct classification fine-tuning baseline.

### 2. Sequence Modeling Architecture

Sequence-to-sequence design notes and environment metadata:

- RNN and LSTM sequence modeling concepts.
- Encoder-decoder Seq2Seq architecture.
- Transformer component design.
- Training utilities and evaluation workflow.

Files:

- `machine_translation/PROJECT_SUMMARY.md`: architecture summary and scope.
- `machine_translation/seq2seq_model.png`: Seq2Seq architecture diagram.
- `machine_translation/environment.yml` and `requirements.txt`: environment metadata.

## Tech Stack

- PyTorch
- HuggingFace Transformers
- HuggingFace Datasets
- torchtext / spaCy

## Installation

```bash
git clone https://github.com/wangwang11112222/deep-learning.git
cd deep-learning
pip install -r requirements.txt
```

## Usage

```bash
python final_project/contextdistillation5.py
python final_project/vanillaFineTune.py
```

## Engineering Focus

This repo highlights model-training pipeline design, teacher/student data flow, GPU-oriented experiment setup, and reproducible model comparison.
