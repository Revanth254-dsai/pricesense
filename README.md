# PriceSense

Estimating what a product costs from its description, using a 4-bit quantised
Llama 3.2 fine-tuned with QLoRA.

A generalist frontier model has to be good at everything. A 1B open model,
tuned on one narrow task, only has to be good at this. That trade is the whole
experiment.

## Design

| Stage | Module | Runs on |
|---|---|---|
| Build prompt/completion pairs | `prepare.py` | laptop |
| Reference baselines | `baselines.py` | laptop |
| QLoRA fine-tune | `train.py` | GPU / Colab |
| Inference | `predict.py` | either |
| Scoring | `evaluate.py` | either |
| Leaderboard | `report.py` | laptop |

Every prompt ends with a dangling `Price is $`, so the only sensible
continuation is digits. During training a completion-only collator uses that
same string as a marker and masks the loss on everything before it - the model
is never rewarded for parroting back the description it was given.

Test completions keep their true decimal price. Rounding the ground truth
would quietly inflate every score.

## Setup

```bash
uv venv
uv pip install -e .
cp .env.example .env   # then fill in HF_USER and HF_TOKEN
```

## Usage

```bash
pricesense prepare --push
pricesense baseline
pricesense eval --base --label "Llama 3.2 1B (base)" -n 100
pricesense train --cap 20000          # on a GPU box
pricesense eval --label "Llama 3.2 1B (QLoRA)"
pricesense report
pricesense price "Bosch 18V cordless drill with two batteries and a case"
```

## Training on Colab

`train.py` refuses to run without CUDA. Push the repo to GitHub, then in a
Colab notebook with a T4 runtime:

```python
!git clone https://github.com/<you>/pricesense.git
%cd pricesense
!pip install -q -e ".[gpu]"

import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_USER"] = "<your-hf-username>"
os.environ["RUN_NAME"] = "pricesense-llama-1b"

!pricesense train --cap 20000
```

The adapter lands on the Hub, and `pricesense eval` picks it up from your
laptop afterwards.