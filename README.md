# QuantProbe

**Quantization-Aware Hallucination Detection in Large Language Models**

Garima Aishwarya

---

## The question

When an LLM is compressed FP16 → INT8 → INT4, does the internal truthfulness
signal - the thing a linear probe reads off the hidden states - survive?

This matters because the deployed model is almost never the FP16 one.
Everyone quantizes for serving. If a hallucination detector is built and
validated on FP16 and then run against an INT4 model, nobody has checked
whether it still works.

### Hypotheses

| | Claim |
|---|---|
| **H1** | Probe accuracy degrades monotonically with bit-width: FP16 ≥ INT8 ≥ INT4. |
| **H2** | Degradation is not uniform across layers. |
| **H3** | A probe trained on FP16 states transfers poorly to INT4 states *even when* an INT4-native probe works fine - the signal is still there, it moved. |
| **H4** | Representational drift predicts probe degradation, layer by layer. |

H3 is the interesting one. If it holds, the finding is "you must recalibrate
your detector after quantizing", which is a concrete, useful claim.

---

## What is actually being measured

`bitsandbytes` INT8/INT4 quantize the **weights**. The compute dtype stays
FP16, so **every hidden state this repo extracts is an FP16 tensor at every
precision setting.** This measures weight-rounding-induced representational
drift, not activation quantization.

This is the first thing a reviewer will ask about. Read
`src/models/loader.py` before changing anything about how precision is applied.

---

## Status

| Phase | State |
|---|---|
| 0 - Scaffold | done |
| 1 - Data loaders + topic splits | done |
| 2 - Extraction (FP16/INT8/INT4) | done - 3 arms, 6049 rows, tokens verified identical |
| 3 - Probes + E1 | done - **H1 rejected: no significant degradation (p=0.38)** |
| 4 - Transfer + drift (E2/E3/E4) | done - **H3 direction right, magnitude ~0.018 AUROC** |
| 5 - Behavioural control (E5) | done - **behaviour flat too; NF4 at 1B changes nothing measurable** |
| 6 - Second model + write-up | done - **Qwen2.5-1.5B replicates; paired McNemar confirms INT8 significant** |

---

## Setup

### Local (CPU) - scaffold, data work, probing

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

`bitsandbytes` will not be usable locally without a CUDA GPU. That is
expected; it is imported lazily and only the INT8/INT4 arms need it.

Point the caches at a drive with room. On the dev laptop `C:` has ~10 GB free
and `D:` has ~147 GB, so:

```bash
source scripts/env.sh          # bash
```

```powershell
. .\scripts\env.ps1            # PowerShell
```

Both set `QUANTPROBE_CACHE`, `HF_HOME`, and pass through `HF_TOKEN` if you
have exported one.

### Colab / Kaggle - extraction

Extraction needs CUDA. Do **not** install `requirements.txt` wholesale on
Colab; it would replace Colab's CUDA torch with a CPU wheel. Instead:

```bash
pip install -q -U transformers==5.16.1 datasets==5.0.1 accelerate==1.14.0 bitsandbytes==0.50.2
```

Set `QUANTPROBE_CACHE` to a mounted Drive path so the extracted activations
survive a session disconnect.

### HuggingFace token

Needed only for gated models (Llama-3.2 is gated). Create a read token at
<https://huggingface.co/settings/tokens>, accept the licence on the model
page, then:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

Never put the token in a YAML file - `src/models/loader.py` reads it from the
environment only, and `.env` / `*.token` are gitignored.

---

## Running things

Every script is CLI-driven off a YAML config. No hardcoded paths, no notebook
as the source of truth.

```bash
# Phase 0: prove the Stage A code path at N=10
python scripts/00_smoke_test.py --config configs/model_smoke.yaml

# The same script against the real model, on a CUDA machine
python scripts/00_smoke_test.py --config configs/model_llama1b.yaml --precision int4

# Phase 1: load all three datasets, print balance + samples, verify splits
python scripts/data_report.py

# Phase 2: extract hidden states. Smoke at N=10 first, always.
python scripts/01_extract.py --config configs/model_smoke.yaml --limit 10
python scripts/01_extract.py --config configs/model_llama1b.yaml   # needs CUDA
python scripts/01_extract.py --config configs/model_llama1b.yaml --verify-only

# Phase 3: fit probes, plot E1. CPU only - reads the cached arrays.
python scripts/02_probe.py --config configs/experiment_e1.yaml
python scripts/02_probe.py --config configs/experiment_e1.yaml --jobs 8

# Phase 4: E2 transfer, E3 drift, E4 rotation. CPU only.
python scripts/03_analyze.py --config configs/experiment_e1.yaml --jobs 6

# Phase 5: E5 behavioural control (needs CUDA) and the abstention demo (CPU).
python scripts/04_behavioural.py --config configs/model_llama1b.yaml --limit 50
python scripts/05_abstention.py --config configs/experiment_e1.yaml

# Tests. Offline by default; -m network hits the real Hub.
python -m pytest tests/ -q
python -m pytest tests/ -q -m network
```

### Verified dataset identifiers

Found by searching the Hub and confirmed against real rows on 2026-09-02.
**Label convention is 1 = TRUE in all three**, established from content, not
from a README.

| Dataset | Repo / config | Records | Positive rate |
|---|---|---|---|
| Azaria & Mitchell true-false | `pminervini/true-false`, 6 topic splits | 6049 after cleaning | 0.504 |
| TruthfulQA MC1 | `truthfulqa/truthful_qa` / `multiple_choice` | 4114 choices from 817 questions | 0.199 |
| HaluEval QA | `pminervini/HaluEval` / `qa` | 2 per source row | 0.500 by construction |

Gotchas that are handled in `src/data/loaders.py`: the
repo's `cieacf` split is the six topics concatenated (loading it too would
duplicate everything); 14 `cities` statements carry both labels and are
dropped; TruthfulQA's topic has to be joined in from the `generation` config.

---

## Layout

```
quantprobe/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml             # everything shared across runs
│   ├── model_smoke.yaml      # SmolLM2-135M, CPU, code-path check only
│   ├── model_llama1b.yaml    # Llama-3.2-1B-Instruct, the v1 model
│   └── experiment_e1.yaml
├── src/
│   ├── config.py             # YAML + `extends:` + ${ENV} interpolation
│   ├── seeding.py            # all four RNGs
│   ├── logging_utils.py      # logger + provenance sidecars
│   ├── paths.py              # env-var-driven path resolution
│   ├── data/loaders.py       # Phase 1 -> unified schema
│   ├── models/loader.py      # precision-aware model loading
│   ├── extract/hidden.py     # Phase 2, Stage A
│   ├── probes/linear.py      # Phase 3, Stage B
│   ├── analysis/transfer.py  # E2
│   ├── analysis/drift.py     # E3, E4
│   └── viz/plots.py
├── scripts/
│   ├── 00_smoke_test.py
│   ├── 01_extract.py
│   ├── 02_probe.py
│   └── 03_analyze.py
├── tests/
├── cache/                    # gitignored: hidden states
└── results/                  # committed: metrics json, figures
```

### Config inheritance

`configs/base.yaml` holds everything that must be identical across every run -
the seed, the token position, the save dtype, the quantization settings.
Model configs `extends: base.yaml` and override only what genuinely differs.
This is not tidiness: if the seed or the token position drifted between two
precision arms, the comparison between them would be meaningless and nothing
would visibly break.

---

## Data

Unified schema across all three datasets:

```json
{ "id": "str", "prompt": "str", "statement": "str",
  "label": 0, "topic": "str", "source": "str" }
```

**Splits are by topic, never random.** Train on 5 topics, test on the held-out
one. Random splits leak near-duplicate statements and inflate AUROC - a known
failure mode in probe papers.

---

## Experiments

| ID | Experiment | Output |
|---|---|---|
| E1 | Per-layer probe AUROC, one curve per precision | headline figure |
| E2 | Cross-precision transfer, all 9 cells | 3×3 heatmap |
| E3 | Geometric drift: cosine / L2 / CKA vs FP16 | drift curve |
| E4 | Probe direction stability: angle between FP16 and INT4 weight vectors | rotation vs noise |
| E5 | Behavioural control: does INT4 actually hallucinate more? | separates "model got worse" from "detector got worse" |

E5 is the control that keeps the result honest. If INT4 task accuracy is
unchanged but probe AUROC drops, the **detector** broke, not the model.

---

## References

- Azaria & Mitchell (2023), *The Internal State of an LLM Knows When It's Lying*
- Burns et al. (2022), *Discovering Latent Knowledge in Language Models Without Supervision*
- Marks & Tegmark (2023), *The Geometry of Truth*
- Dettmers et al. (2022), *LLM.int8()*
- Dettmers et al. (2023), *QLoRA* - NF4
