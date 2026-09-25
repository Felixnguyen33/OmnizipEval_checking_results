# WorfBench

Node-level workflow evaluation for the eight WorfBench tasks used in this check:
AlfWorld, InterCodeSQL, Lumos, OS, Seal-Tools, ToolAlpaca, ToolBench and WebShop.

The benchmark code in this folder is [zjunlp/WorfBench](https://github.com/zjunlp/WorfBench)
(`node_eval.py`, `evaluator/`, `prompts/`, `gold_traj/`, `LLM/`). The upstream
writeup is in [WORFBENCH.md](WORFBENCH.md). `evaluate.py` generates workflows
with vLLM chat at temperature 0 and scores node precision, recall and F1 with
`sentence-transformers/all-mpnet-base-v2`.

InterCodeSQL and Seal-Tools have no two-shot examples in the benchmark, so those
two tasks are zero-shot. The other six tasks use the bundled two-shot prompts.
Unparsed workflows count as zero. The reported average weights the eight tasks
equally.

Set `HF_HOME` to the Hugging Face cache you want to use. Write `--out-root`
outside this repository; predictions and scores are not part of the source tree.

```bash
pip install -r worfbench/requirements.txt
CUDA_VISIBLE_DEVICES=0 python worfbench/evaluate.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --out-root /path/to/worfbench_outputs
```

`python worfbench/evaluate.py --help`
