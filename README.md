# PriceSense

Estimating what a product costs from its description, using a 4-bit quantised
Qwen 2.5 fine-tuned with QLoRA.

A generalist model has to be good at everything. A 0.5B open model, tuned on one
narrow task, only has to be good at this. That trade is the experiment.

## Result

Fine-tuning cut mean absolute error by **63%** — from $221.90 to $81.95 — using a
**26 MB** LoRA adapter against a 988 MB base model. The same 0.5B model, the same
prompts; only the adapter differs.

| Model | MAE | RMSE | r² | Within 20%/$40 |
|---|---:|---:|---:|---:|
| **Qwen 2.5 0.5B (QLoRA)** | **$81.95** | $144.35 | **+15.2%** | 48% |
| Word count linear regression | $105.73 | $148.17 | 0.1% | 20.5% |
| Constant (training mean) | $106.09 | $148.35 | -0.1% | 18.5% |
| Llama 3.2 1B (zero-shot) | $122.68 | $257.88 | -170.8% | 49% |
| Llama 3.2 3B (zero-shot) | $130.22 | $240.25 | -135.0% | 41% |
| Qwen 2.5 0.5B (base) | $221.90 | $438.56 | -683.2% | 36% |

*n=100 held-out test items. Zero-shot models served locally through Ollama.*

Two things worth pulling out of that table.

**Every model except the fine-tune has a negative r².** They are all worse than
simply predicting the training mean for every item. Understanding a product and
knowing its price turn out to be different skills.

**The fine-tuned 0.5B beats zero-shot Llama 3.2 3B by $48** — six times the
parameters, no contest. Scale does not substitute for calibration on a narrow task.

## How it works

Each product becomes a prompt that dead-ends on a dollar sign:

```
What does this cost to the nearest dollar?

Title: Excess V2 Distortion/Modulation Pedal
Category: Music Pedals
Brand: Old Blood Noise
Description: A versatile pedal offering distortion and three modulation modes...

Price is $
```

The only sensible continuation is digits — the model has no room to write prose.
That same `Price is $` string is then used by a completion-only collator during
training, which masks the loss on everything before it. The model gets no credit
for reproducing the description it was handed; all of its capacity goes into the
number.

Training is QLoRA: the base model is loaded in 4-bit NF4 (451 MB resident), frozen
entirely, and small low-rank adapters are trained on the attention and MLP
projections. Roughly 1% of parameters are updated.

## Design

| Stage | Module | Runs on |
|---|---|---|
| Build prompt/completion pairs | `prepare.py` | laptop |
| Reference baselines | `baselines.py` | laptop |
| Locally-served LLM baselines | `ollama_backend.py` | laptop |
| QLoRA fine-tune | `train.py` | GPU / Colab |
| Inference | `predict.py` | either |
| Scoring | `evaluate.py` | either |
| Leaderboard | `report.py` | laptop |

The scorer takes any `Callable[[str], float]`, so a constant baseline, a linear
regression, an Ollama HTTP call and a quantised local transformer all go through
identical evaluation code. Scorecards persist to `results/*.json` and the
leaderboard assembles itself from whatever is there.

Everything runs on a laptop except training, which needs CUDA. `train.py` is a CLI
command rather than a notebook precisely so it runs unmodified on Colab, a
cluster, or anything else with a GPU.

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
pricesense ollama --model llama3.2:1b --label "Llama 3.2 1B (zero-shot)" -n 100
pricesense eval --base --label "Qwen 0.5B (base)" -n 100
pricesense eval --adapter <user>/pricesense-qwen-0.5b --label "Qwen 0.5B (QLoRA)" -n 100
pricesense report
pricesense price "Bosch 18V cordless drill with two batteries and a case"
```

## Training

On Colab with a T4 runtime:

```python
!git clone https://github.com/<you>/pricesense.git
%cd pricesense
!pip install -q -e ".[gpu]"

import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_USER"] = "<your-hf-username>"
os.environ["RUN_NAME"] = "pricesense-qwen-0.5b"

!pricesense train
```

20,000 examples, 5,000 steps, one epoch, ~56 minutes. The adapter is pushed to the
Hub and picked up by `pricesense eval` afterwards.

## Data

`ed-donner/items_lite` — 20,000 train / 1,000 validation / 1,000 test product
listings with real prices.

Descriptions are truncated at **110 tokens**, measured on the Qwen tokenizer. That
clips 12.9% of items; the resulting sequences top out at 129 tokens, which is what
`max_seq_len` is set from. Train and validation completions are rounded to whole
dollars so the model learns a clean output shape. **Test completions keep their
true decimal price** — rounding the ground truth would quietly inflate every score.

The token cutoff is tokenizer-specific. Changing the base model means re-running
`prepare` before training.

## What the model gets wrong

The base model's failures are unbounded and upward: **$1,999 for a stick of RAM
that costs $124**, $1,299 for a $76 shower faucet. It has one idea — "technical
product, four digits."

After fine-tuning the errors invert. The model now under-predicts expensive items:
$199 for an $819 DSLR bundle, $150 for a $713 violin, $110 for a $475 faucet. It
learned that most items in this catalogue are cheap and regresses toward that.

This is a healthier failure mode — bounded rather than catastrophic — and it is the
expected signature of training on a right-skewed target with a symmetric loss. It
is also the clearest direction for improvement: the model needs a reason to risk a
large estimate.

## Honest limitations

- **n=100 test sample.** Enough to separate $82 from $222; not enough to split
  hairs between adjacent bars. An earlier run at n=50 flipped the 1B/3B ordering.
- **One epoch.** Training loss fell from 1.159 to 0.960, with most of the drop in
  the first 200 steps and a long flat tail. Loss was a poor guide here — it barely
  moved while dollar error dropped a third. The training metric is not the
  business metric.
- **Zero-shot comparisons are not controlled.** Llama 3.2 1B/3B are instruction-
  tuned models of different architecture and size; they set a bar rather than
  isolate a variable. The base-vs-fine-tuned Qwen pair is the controlled result.
- **A probability-weighted decoding variant was tried and removed.** It averaged
  the top-k *first* tokens, which works on tokenizers that group digits but
  collapses on Qwen, where numbers tokenize digit by digit. It produced estimates
  under $2 for $800 items. Greedy decoding is used throughout.

## Next

- Fine-tune Llama 3.2 1B for a genuinely controlled comparison against its own
  zero-shot number ($122.68).
- Weighted or log-space loss to address the under-prediction on expensive items.
- Second epoch — loss plateaued but showed no overfitting.