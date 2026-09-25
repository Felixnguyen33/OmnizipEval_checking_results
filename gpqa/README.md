# GPQA and GPQA Diamond

Scripts in this folder are the launchers that produced the OmniZipEval GPQA
runs. Run them from this folder. Outputs go to `outputs/` and are not part of
this repository.

```text
scripts/eval/gpqa.sh
scripts/eval/run_gpqa_qwen25_7b_instruct.sh
scripts/eval/run_gpqa_deepseek_r1_llama8b.sh
scripts/eval/run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh
scripts/eval/run_gpqa_deepseek_r1_smoothquant_qwen25_protocol.sh
scripts/eval/run_gpqa_pruned_sq_qwen25_protocol.sh
scripts/eval/deepseek_llama_8b_gpqa_examples.sh
scripts/eval/tasks/gpqa_qwen25/
scripts/eval/tasks/gpqa_deepseek_r1/
```

Dataset: `Idavidrein/gpqa`. Main is 448 questions. Diamond is 198 questions.
The saved runs used `lm-eval` 0.4.11 and the vLLM backend, chat template on,
no system prompt, seeds `random=0`, `numpy=1234`, `torch=1234`, `fewshot=1234`.

## Two protocols

**Qwen2.5 Instruct** (`run_gpqa_qwen25_7b_instruct.sh` and the `*_qwen25_protocol.sh` scripts):

- tasks `gpqa_main_qwen25_cot_zeroshot` and `gpqa_diamond_qwen25_cot_zeroshot`
- 0-shot, greedy (`temperature=0`, `do_sample=False`), one sample
- `max_gen_toks=8192`, `max_model_len=16384`
- prompt ends with `Let's think step by step:`
- score `exact_match,strict-match` after extracting the last `\boxed{}`, `Answer:`, `The answer is`, or `(A)`–`(D)` following `</think>` when present

**DeepSeek-R1 report** (`run_gpqa_deepseek_r1_llama8b.sh`):

- tasks `gpqa_main_deepseek_r1_cot` and `gpqa_diamond_deepseek_r1_cot`
- 0-shot, `temperature=0.6`, `top_p=0.95`, `do_sample=True`
- prompt asks for `\boxed{}` and prefixes `<think>`
- report Average Pass@1 uses 64 samples; the launcher default is `N_SAMPLES=1`

`gpqa.sh` is a separate 5-shot greedy sweep (`gpqa_diamond_n_shot` by default).
`deepseek_llama_8b_gpqa_examples.sh` is a 5-shot sampled `gpqa` group command.
Neither of those two produced the complete main/diamond result files checked below.

## What the saved result files used

Twenty complete result files were checked (10 models × main and/or diamond).
Nineteen of them are the Qwen2.5 greedy protocol, one sample, full split
(448 or 198). One file is the DeepSeek-R1 Diamond task with sampling turned on
and `repeats=1`, so it is a single sample, not Average Pass@1 over 64.

| Checkpoint | Split | Task | strict-match or pass@1 |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | main | qwen25 | 33.93 |
| Qwen2.5-7B-Instruct | diamond | qwen25 | 33.84 |
| DeepSeek-R1-Distill-Llama-8B | main | qwen25 | 31.25 |
| DeepSeek-R1-Distill-Llama-8B | diamond | qwen25 | 26.77 |
| DeepSeek-R1-Distill-Llama-8B | diamond | stock cot_zeroshot | 0.00 strict / 25.25 flexible |
| DeepSeek-R1-Distill-Llama-8B | diamond | deepseek, N=1 | 42.93 pass@1 |
| Llama-8B SmoothQuant W8A8 | main / diamond | qwen25 | 28.57 / 31.82 |
| Qwen-7B SmoothQuant W8A8 | main / diamond | qwen25 | 34.38 / 32.83 |
| Qwen3-8B SmoothQuant W8A8 | main / diamond | qwen25 | 47.99 / 49.49 |
| deepseek_qwen_7b_wanda_50 | main / diamond | qwen25 | 17.63 / 14.14 |
| deepseek_qwen_7b_gblm_50 | main / diamond | qwen25 | 14.06 / 13.13 |
| deepseek_qwen_7b_sparsegpt_50 | main / diamond | qwen25 | 12.05 / 11.62 |
| deepseek_qwen_1_5b_gblm_50 | main / diamond | qwen25 | 5.13 / 4.55 |

Scores are percentages. The stock `gpqa_diamond_cot_zeroshot` row only accepts
`The answer is (X)`. DeepSeek-R1 answers in `\boxed{}` were scored as misses
(`strict-match=0`). The Qwen2.5 task files exist to avoid that miss.

## What is not comparable

- Instruct and distilled checkpoints in the table above share the greedy
  Qwen2.5 task. That is one protocol. It is not the DeepSeek-R1 report
  protocol (temperature 0.6, 64 samples, boxed prompt). The single sampled
  Diamond file (42.93) cannot be averaged with the greedy files.
- `N_SAMPLES` defaults to 1 in `run_gpqa_deepseek_r1_llama8b.sh`. The report
  number needs `N_SAMPLES=64`.
- Choice order is `random.shuffle` inside `process_docs`, using the process
  RNG after `random_seed=0`. A different harness version or an extra RNG call
  before document processing changes which letter is correct.
- On several Qwen2.5-protocol files, `strict-match` and `flexible-extract`
  disagree (Qwen3-8B SmoothQuant Diamond 49.49 vs 39.39). Report one filter
  for every row.
- These scripts record the Hub id that was loaded. They do not check that a
  pruned checkpoint's parent weights match the dense model it is compared with.

## Run

```bash
python -m pip install -r requirements.txt

# Qwen2.5-7B-Instruct, main and diamond, greedy 0-shot
GPUS=0,1,2,3 bash scripts/eval/run_gpqa_qwen25_7b_instruct.sh

# DeepSeek-R1 Diamond, report sampling; set 64 for Average Pass@1
ONLY=diamond N_SAMPLES=64 GPUS=0,1,2,3 \
  bash scripts/eval/run_gpqa_deepseek_r1_llama8b.sh

# Same greedy Qwen2.5 task on a DeepSeek-R1 checkpoint
GPUS=0,1,2,3 bash scripts/eval/run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh
```

`ONLY=main`, `ONLY=diamond`, or `ONLY=both`. `LIMIT` truncates the split.
Rerun every model you compare with the same script, GPU count, and filter.
