# OmnizipEval_checking_results

Portable scripts for checking GSM8K and TruthfulQA evaluation consistency.
No historical logs, result files, model weights, credentials, or machine-specific
model paths are included. Model and dense-parent tokenizer inputs are explicit.

## What was inconsistent in the original evaluation?

- Some Qwen2.5 pruned checkpoints had a base-model parent but were compared with
  an Instruct baseline. Selecting the same tokenizer does not fix a weight-parent mismatch.
- DeepSeek-Llama compressed GSM8K responses contained undecoded byte-level token
  markers; its dense responses did not. This changed answer extraction.
- A DeepSeek-Llama GSM8K checkpoint labeled GPTQ-Int4 was actually eight-bit.
- Historical runs mixed prompting, scoring, model identities and execution settings.
- The corrected Qwen2.5-3B SparseGPT TruthfulQA pair and corrected 3B pruning GSM8K
  cohort have stronger matching evidence. Qwen3 MoE TruthfulQA scores reproduce
  with matching documents and prompts, although hardware parallelism differs.

Before export, saved scores were independently recalculated for 34 complete GSM8K
runs and 18 TruthfulQA task outputs. All reproduced under their respective original
scorers. Reproducibility alone does not establish fairness.

## Files

- `evaluate.py`: one Hugging Face backend for both benchmarks; explicit checkpoint
  and parent tokenizer; pinned Hub revisions; actual checkpoint quantization metadata.
- `tasks/gsm8k_checked/`: fixed four-example chat task and strict boxed-answer scorer.
- `check_pair.py`: independently rescores two output directories and compares
  documents, rendered prompts, generation arguments, package versions and protocol.
- `test_scoring.py`: regression tests for decoding and final-answer extraction.

## Protocol (checked_v1)

Both datasets use BF16 computation, float32 likelihood normalization, seed 0, no
response/request caching, an 8,192-token context, and batch size 1 by default.
Use the same environment, hardware arrangement and batch size for each paired run.
The model must be compatible with the HF backend; no backend fallback is automatic.

**GSM8K:** all 1,319 test questions, pinned dataset revision, four fixed demonstrations
as separate chat turns, greedy decoding, maximum 4,096 generated tokens. Thinking
is disabled through the tokenizer option where supported; some R1 models still emit
reasoning. The scorer uses the last complete numeric `\boxed{...}` after `</think>`
when present. It accepts signed integers/decimals with optional grouping commas or
a dollar sign. Fractions, expressions, prose-only answers, incomplete boxes and
unfinished explicitly marked reasoning receive zero. Raw byte-level token markers
raise an error. There is no fallback to intermediate reasoning numbers.

**TruthfulQA:** all 817 validation questions, pinned dataset revision, standard
lm-eval MC1/MC2 likelihood tasks, no chat template and zero additional few-shot
examples. The standard prompt still contains six fixed demonstrations. MC1 is
single-answer accuracy; MC2 is normalized probability mass on true candidates.

This is a **new protocol**. In particular, GSM8K strict boxed scoring and the HF
backend are not identical to the historical custom fallback scorer/vLLM runs.
Rerun the dense baseline and every variant together; do not combine new scores
with old tables. No new GPU evaluations or accuracy claims are included here.

## Run

Use Python 3.10+ and a compatible GPU/PyTorch installation:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_scoring.py
python evaluate.py --help
```

Quantized checkpoints may require their format-specific HF dependencies. Unsupported
formats should fail rather than silently switch precision or backend. Package versions
are recorded in each output manifest; reuse the exact same environment for a cohort.

Example with user-provided local directories (replace the placeholders):

```bash
python evaluate.py --dataset gsm8k \
  --model /path/to/dense-parent \
  --parent-tokenizer /path/to/dense-parent \
  --output outputs/gsm8k/base

python evaluate.py --dataset gsm8k \
  --model /path/to/compressed-checkpoint \
  --parent-tokenizer /path/to/dense-parent \
  --output outputs/gsm8k/variant

python check_pair.py outputs/gsm8k/base outputs/gsm8k/variant
```

For TruthfulQA, change `--dataset truthfulqa` and use new output folders. For Hub
IDs, provide `--revision` and `--parent-revision`, each a full immutable commit SHA.
For models requiring sharding, use `--parallelize` consistently across the cohort.
Output directories must be new; existing outputs are never overwritten.

## Limits of an automated fairness check

An exit code of zero from `check_pair.py` means the saved scores reproduce and the
observed dataset/prompts/protocol match. It is not a fairness certificate. Verify the
actual dense weight parent, pruning calibration and sparsity, quantization bit width,
and generation truncation separately. Local paths/config hashes are not full weight
hashes. The manifest explicitly records `parent_identity_verified: false`.

Outputs and samples are generated locally for verification and ignored by Git.
No existing source workspace artifacts were deleted to prepare this repository.
