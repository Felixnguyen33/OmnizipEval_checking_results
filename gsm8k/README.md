# GSM8K

This folder is self-contained. `evaluate.py` runs all 1,319 test questions using
four fixed demonstrations as separate chat turns, greedy decoding and a maximum
of 4,096 generated tokens. The dataset revision is pinned in `task.yaml`.

The strict scorer uses the last complete numeric `\boxed{...}` after `</think>`
when present. It accepts signed integers/decimals, grouping commas and an optional
dollar sign. Fractions, expressions, prose-only answers, incomplete boxes and
unfinished explicitly marked reasoning receive zero. Raw byte-level token markers
raise an error. It never falls back to intermediate reasoning numbers.

Thinking is disabled through the tokenizer option where supported; some R1 models
still emit reasoning. This is a new protocol, not a reproduction of historical
fallback-scored vLLM accuracy. Rerun dense and compressed models together.

## Run

Use Python 3.10+ with a GPU-compatible PyTorch installation. Run these commands
from this folder, or prefix script paths with the dataset folder from the repository root.

```bash
python -m pip install -r requirements.txt
python evaluate.py --help

python evaluate.py \
  --model /path/to/dense-parent \
  --parent-tokenizer /path/to/dense-parent \
  --output outputs/base

python evaluate.py \
  --model /path/to/compressed-checkpoint \
  --parent-tokenizer /path/to/dense-parent \
  --output outputs/variant

python check_pair.py outputs/base outputs/variant
```

For Hub IDs, supply `--revision` and `--parent-revision`, each a full immutable
commit SHA. Local paths are supplied by the caller. For models requiring sharding,
use `--parallelize` consistently. Existing output directories are refused.

Use the same environment, hardware arrangement and batch size for each pair.
The HF backend uses BF16 computation, float32 likelihood normalization, seed 0,
an 8,192-token context, and no response/request cache. Quantized checkpoints may
require format-specific dependencies; the actual quantization configuration is
recorded without assuming a bit width.

`check_pair.py` rescores the saved responses and compares documents, prompts,
generation arguments and protocol metadata. Success does not verify weight-parent
identity, calibration fairness, actual sparsity, or truncation. Check those separately.
Outputs are generated locally and excluded from Git. No new GPU scores are supplied.

## Scorer tests

```bash
python -m unittest -v test_scoring.py
```
