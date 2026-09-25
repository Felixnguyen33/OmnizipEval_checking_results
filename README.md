# OmnizipEval_checking_results

Each dataset has its own runner, consistency checker and installation instructions:

- [gsm8k/](gsm8k/README.md) — four-shot chat evaluation and strict boxed-answer scoring.
- [truthfulqa/](truthfulqa/README.md) — MC1/MC2 likelihood evaluation.
- [gpqa/](gpqa/README.md) — GPQA main and GPQA Diamond. Shared runs use `lm-eval` 0.4.9.1 and task version 2.0.
- [worfbench/](worfbench/README.md) — eight-task WorfBench node precision, recall and F1.

```text
gsm8k/
  evaluate.py
  check_pair.py
  task.yaml
  utils.py
  test_scoring.py
  requirements.txt
  README.md
truthfulqa/
  evaluate.py
  check_pair.py
  requirements.txt
  README.md
gpqa/
  README.md
  requirements.txt
  scripts/eval/          # launchers and task YAMLs actually used
worfbench/
  evaluate.py
  node_eval.py
  evaluator/
  prompts/
  gold_traj/
  README.md
```

For example, run `python gsm8k/evaluate.py --help` or
`python truthfulqa/evaluate.py --help`. No `--dataset` flag is needed.
Both folders can also be used independently.

The repository contains source code and instructions only: no historical logs,
results, weights, credentials or machine-specific model paths.

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

Twenty complete GPQA result files (main and Diamond) were checked the same way.
Nineteen used the Qwen2.5 greedy 0-shot task. One Diamond file used DeepSeek-R1
sampling with a single draw, and one earlier Diamond file used the stock
`The answer is` parser, which scored boxed answers as zero. Details are in
[gpqa/README.md](gpqa/README.md).

## Comparison policy

These scripts use the `checked_v1` protocol. GSM8K uses a different scorer/backend
from the historical fallback-scored vLLM results. Rerun the baseline and variants
together; do not mix new scores with old tables. Matching scripts alone cannot fix
a base-versus-Instruct checkpoint mismatch. Dataset-specific details are in each folder.
