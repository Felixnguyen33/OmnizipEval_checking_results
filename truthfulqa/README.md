# TruthfulQA

This folder is self-contained. `evaluate.py` runs standard lm-eval MC1 and MC2
likelihood tasks on all 817 validation questions using a pinned dataset revision.
There is no chat template and no additional few-shot sampling. The standard task
prompt still includes six fixed demonstrations.

MC1 measures single-answer accuracy; MC2 measures normalized probability mass on
true candidates. Both are reported on the 0–1 scale by the evaluator and as
percentages by `check_pair.py`. Task definitions come from the pinned lm-eval
package; their file hashes are recorded with each run.

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
