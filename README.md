# Deep Learning Projects

Two deep-learning research projects covering **context distillation** for LLMs and **sequence modeling** for machine translation.

## Projects

### 1. Context Distillation (Final Project)

Implements the context distillation paradigm from *"Teaching Small Language Models to Reason"* (Fu et al., 2022):

- **Teacher model**: OPT-125M / OPT-350M (causal LM) generates reasoning chains via prompt-based sentiment analysis
- **Answer extraction**: Heuristic label extraction from teacher outputs
- **Student model**: OPT-125M / OPT-350M fine-tuned as a sequence classifier on extracted labels (SST-2 dataset)
- **Baseline comparison**: Vanilla fine-tuning without teacher-generated reasoning

```
Teacher (Causal LM)                    Student (Classifier)
     │                                       │
     ▼                                       ▼
Prompt: "Analyze sentiment..."        Input: Same prompt
     │                                       │
     ▼                                       ▼
Generated reasoning chain              Trained on teacher labels
     │                                       │
     ▼                                       ▼
Extracted label (0/1)                  Inference: direct classification
```

**Files:**
- `final_project/contextdistillation5.py` — Context distillation pipeline (teacher generation → label extraction → student training)
- `final_project/vanillaFineTune.py` — Baseline vanilla fine-tuning without reasoning chains

### 2. Machine Translation (Sequence Modeling)

Coursework covering sequence-to-sequence modeling (summary only — implementation files excluded due to redistribution restrictions):

- Naive RNN and LSTM implementations
- Encoder-decoder Seq2Seq architecture
- Transformer model components
- Training utilities and evaluation workflow

**Files:**
- `machine_translation/PROJECT_SUMMARY.md` — Topic summary and scope
- `machine_translation/seq2seq_model.png` — Seq2Seq architecture diagram
- `machine_translation/environment.yml` / `requirements.txt` — Conda/pip environment

## Project Structure

```
deep-learning/
├── final_project/
│   ├── contextdistillation5.py    # Context distillation experiment
│   ├── vanillaFineTune.py         # Baseline fine-tuning
│   └── mainPart.txt               # LaTeX figure template (results)
├── machine_translation/
│   ├── PROJECT_SUMMARY.md         # Coursework summary
│   ├── seq2seq_model.png          # Architecture diagram
│   ├── environment.yml            # Conda environment
│   └── requirements.txt           # Pip dependencies
├── requirements.txt               # Root-level dependencies
├── .gitignore
└── README.md
```

## Tech Stack

- **PyTorch** — Model implementation and training
- **HuggingFace Transformers** — OPT-125M/350M, AutoModelForCausalLM, AutoModelForSequenceClassification
- **HuggingFace Datasets** — GLUE SST-2 sentiment benchmark
- **torchtext / spaCy** — Tokenization for machine translation

## Installation

```bash
git clone https://github.com/wangwang11111222/deep-learning.git
cd deep-learning
pip install -r requirements.txt
```

> **Note**: Context distillation experiments were originally run on Google Colab with GPU. Remove `!pip install` lines if running locally.

## Usage

```bash
# Context distillation experiment
python final_project/contextdistillation5.py

# Baseline vanilla fine-tuning
python final_project/vanillaFineTune.py
```

## Key Results

The project compared context distillation vs. vanilla fine-tuning across OPT-125M and OPT-350M:

| Model | Method | Training Data | Notes |
|-------|--------|---------------|-------|
| OPT-125M | Context Distillation | 10 teacher-labeled examples | Teacher generates reasoning → student learns from labels |
| OPT-125M | Vanilla Fine-Tuning | 10 ground-truth examples | Direct classification training |
| OPT-350M | Context Distillation | 10 teacher-labeled examples | Larger teacher model |
| OPT-350M | Vanilla Fine-Tuning | 10 ground-truth examples | Direct classification baseline |

## License

MIT
