# Deep Learning

Two deep-learning coursework projects: a **context-distillation** experiment that
transfers reasoning-style supervision from a teacher language model to a smaller
student classifier on SST-2, and a **machine translation** sequence-modeling project
that is included here as documentation only.

## Overview

### 1. Context distillation (`final_project/`)

The experiment asks whether a causal language model can act as a labeller for a
sentiment classifier. A teacher model is prompted to produce a free-text sentiment
analysis of each sentence; a heuristic extractor converts that text into a binary
label; the student model is then fine-tuned as a sequence classifier on those
teacher-derived labels. A baseline script fine-tunes the same student directly on
the ground-truth SST-2 labels, so the two training signals can be compared under an
otherwise identical setup.

Both scripts use `facebook/opt-125m` for the teacher and the student, load SST-2
from the GLUE benchmark via HuggingFace `datasets`, and train on a deliberately
small slice of the data (the first 10 training examples, evaluated on 20 validation
examples) for 15 epochs with AdamW. They evaluate the student before and after
training and plot the per-epoch training loss.

### 2. Machine translation (`machine_translation/`)

Sequence-to-sequence machine translation covering RNN/LSTM design, encoder-decoder
Seq2Seq, and Transformer components. **The implementation source and datasets are
not included in this repository** - only the project summary, the environment
specifications, and the architecture diagram are published here. See
[Notes](#notes).

## Features

- Teacher/student **context-distillation pipeline**: prompt construction, free-text
  generation, heuristic answer extraction, and student fine-tuning.
- **Matched baseline** (`vanillaFineTune.py`) that isolates the effect of
  teacher-derived labels by holding the model, data slice, prompt template,
  optimizer, and epoch count constant.
- Pre-training and post-training evaluation passes reporting average loss and
  accuracy on the same validation slice.
- Training-loss curves plotted with matplotlib.
- Automatic CUDA device selection with CPU fallback.
- Reproducible environment specifications for the machine-translation project
  (`environment.yml` for conda, `requirements.txt` for pip).

## Directory Structure

```text
deep-learning/
|-- final_project/
|   |-- contextdistillation5.py    # Teacher generation + label extraction + student fine-tuning
|   `-- vanillaFineTune.py         # Baseline: direct fine-tuning on ground-truth SST-2 labels
|-- machine_translation/
|   |-- PROJECT_SUMMARY.md         # Scope and architecture summary (docs only)
|   |-- environment.yml            # Conda environment spec (python 3.8.10, torch, torchtext, spacy)
|   |-- requirements.txt           # Pip equivalent
|   `-- seq2seq_model.png          # Seq2Seq architecture diagram
|-- requirements.txt               # Dependencies for the context-distillation scripts
|-- LICENSE
`-- README.md
```

## Installation

```bash
git clone https://github.com/wangwang11111222/deep-learning.git
cd deep-learning

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is optional. Both scripts fall back to CPU, which is workable
here only because the data slice is very small.

For the machine-translation environment specification (documentation only, since the
source is not included):

```bash
conda env create -f machine_translation/environment.yml
conda activate cs7643-a4
```

## Usage

```bash
python final_project/vanillaFineTune.py
```

`vanillaFineTune.py` runs as-is. It downloads `facebook/opt-125m` and the GLUE SST-2
dataset on first run, prints evaluation loss and accuracy before and after training,
and opens a matplotlib window with the training-loss curve.

```bash
python final_project/contextdistillation5.py
```

`contextdistillation5.py` **will not run unmodified.** It was exported from a Google
Colab notebook and still contains notebook shell magics at the top of the file:

```python
!pip install transformers datasets
!pip install accelerate -U
```

Under a normal Python interpreter these lines raise a `SyntaxError` before any other
code executes. To run the script, either remove those two lines (the packages are
already covered by `requirements.txt`) or execute the file in a notebook environment
such as Colab or Jupyter, where the magics are valid. This cleanup is being handled
separately and the lines are left in place here so the file matches its original
exported state.

Both scripts hard-code their configuration - model name, the 10-example training
slice, the 20-example evaluation slice, batch size 2, learning rate 5e-5, weight
decay 1e-4, and 15 epochs. Change them by editing the source; there is no CLI.

## Dependencies

From `requirements.txt`:

- `torch`, `transformers`, `datasets`, `accelerate`
- `matplotlib`, `tqdm`, `numpy`
- `torchtext`, `spacy` (listed for the machine-translation project)

From `machine_translation/environment.yml`: Python 3.8.10, `torch`, `torchtext`,
`spacy`, `numpy`, `matplotlib`, `tqdm`, `jupyter`, `notebook`.

Note that `AdamW` is imported from `transformers` in both scripts. That import is
deprecated in recent `transformers` releases and has been removed in the newest
versions; pin an older `transformers` or switch to `torch.optim.AdamW` if the import
fails.

## Notes

- **`machine_translation/` is documentation only.** The model implementation source
  files and the translation datasets are deliberately not included in this
  repository. The directory holds the project summary, environment specifications,
  and architecture diagram. The empty `models/naive/` and `models/seq2seq/`
  directories are placeholders where that source would live, and are not tracked by
  git. Nothing in this directory is runnable on its own.
- **No results are published here.** The scripts print accuracy and loss when run,
  but no measured numbers, saved checkpoints, or result tables are included in this
  repository, and none should be inferred from this README.
- The teacher label extractor is a deliberately simple heuristic: it marks a
  sentence positive if the teacher's generated text contains any of the words
  "good", "right", or "best", and negative otherwise. It is a coarse proxy, not a
  calibrated parser.
- The training and evaluation slices are very small (10 and 20 examples). They are
  sized for a fast pipeline demonstration, not for statistically meaningful
  comparison between the two training regimes.
- Datasets and pretrained weights are downloaded at runtime from the HuggingFace
  Hub; neither is vendored into this repository.

## License

Released under the [MIT License](LICENSE).
