#!/usr/bin/env bash
# Reproduce DeepSeek-R1 technical-report GPQA numbers (Average Pass@1).
#
# Report targets (arXiv:2501.12948 / DeepSeek-R1 README, Distilled Model Evaluation):
#   DeepSeek-R1-Distill-Llama-8B  · GPQA Diamond pass@1 ≈ 49.0
#   DeepSeek-R1-Distill-Qwen-7B   · GPQA Diamond pass@1 ≈ 49.1
#
# Official protocol this script encodes:
#   - 0-shot generative MCQ (NOT few-shot / NOT greedy loglikelihood)
#   - max_gen_toks = 32768
#   - temperature = 0.6, top_p = 0.95, do_sample = True
#   - N samples / question → Average Pass@1  (default N_SAMPLES=1; report uses 64)
#   - prompt: "Please reason step by step, and put your final answer within \boxed{}."
#   - no system prompt; apply_chat_template; gen_prefix "<think>\n"
#
# Usage:
#   bash scripts/eval/run_gpqa_deepseek_r1_llama8b.sh
#   GPUS=0,1,2,3 bash scripts/eval/run_gpqa_deepseek_r1_llama8b.sh
#   # report-faithful (slow):
#   ONLY=diamond N_SAMPLES=64 bash scripts/eval/run_gpqa_deepseek_r1_llama8b.sh
#   # quick smoke:
#   ONLY=diamond LIMIT=8 bash scripts/eval/run_gpqa_deepseek_r1_llama8b.sh
#   # Qwen distill (report 49.1 on Diamond):
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B ONLY=diamond \
#     bash scripts/eval/run_gpqa_deepseek_r1_llama8b.sh
#
# Requires:
#   pip install lm-eval
#   # recommended backend:
#   pip install vllm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
# Llama-3.1-8B / Qwen2.5-7B heads are divisible by 1/2/4/8 — not by 3/5/6.
# lm-eval's vLLM backend uses tensor_parallel_size=NUM_GPUS (no PP), so pick 1/2/4/8 GPUs.
GPUS="${GPUS:-0,1,2,3}"
resolve_gpus
check_lm_eval

case "$NUM_GPUS" in
  1|2|4|8) ;;
  *)
    die "NUM_GPUS=$NUM_GPUS is not a valid vLLM TP size for this model (use 1, 2, 4, or 8 GPUs).
Example: GPUS=0,1,2,3 bash $0"
    ;;
esac
TP_SIZE="$NUM_GPUS"
PP_SIZE=1

# ---- report defaults ---------------------------------------------------------
N_SAMPLES="${N_SAMPLES:-1}"           # default 1 for fast runs; report uses 64
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
NUM_FEWSHOT=0                         # forced 0-shot (report)
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BACKEND="${BACKEND:-vllm}"            # vllm | hf
ONLY="${ONLY:-diamond}"               # diamond | main | both  (report table = diamond)
LIMIT="${LIMIT:-}"                    # optional lm-eval --limit
BATCH_SIZE="${BATCH_SIZE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

TAG="$(safe_name "$(model_tag "$MODEL")")"
BASE_OUT="${OUT_DIR:-$EVAL_OUTPUT_ROOT/gpqa_deepseek_r1/${TAG}}"
ensure_dir "$BASE_OUT"

TASK_SRC="$SCRIPT_DIR/tasks/gpqa_deepseek_r1"
TASK_RUNTIME="$BASE_OUT/tasks_n${N_SAMPLES}"
rm -rf "$TASK_RUNTIME"
mkdir -p "$TASK_RUNTIME"
cp "$TASK_SRC/utils.py" "$TASK_RUNTIME/"
cp "$TASK_SRC/gpqa_diamond_deepseek_r1_cot.yaml" "$TASK_RUNTIME/"
cp "$TASK_SRC/gpqa_main_deepseek_r1_cot.yaml" "$TASK_RUNTIME/"

# Patch repeats + sampling into the included base yaml for this run.
python - "$TASK_SRC/_gpqa_deepseek_r1_cot.yaml" "$TASK_RUNTIME/_gpqa_deepseek_r1_cot.yaml" \
  "$N_SAMPLES" "$TEMPERATURE" "$TOP_P" "$MAX_GEN_TOKS" <<'PY'
import pathlib, sys
src, dst, n, temp, top_p, max_toks = sys.argv[1:7]
text = pathlib.Path(src).read_text()
text = text.replace("repeats: 64", f"repeats: {int(n)}", 1)
# Keep yaml generation_kwargs in sync with CLI overrides.
import re
text = re.sub(r"temperature:\s*[0-9.]+", f"temperature: {temp}", text, count=1)
text = re.sub(r"top_p:\s*[0-9.]+", f"top_p: {top_p}", text, count=1)
text = re.sub(r"max_gen_toks:\s*[0-9]+", f"max_gen_toks: {int(max_toks)}", text, count=1)
pathlib.Path(dst).write_text(text)
print(f"wrote {dst} (repeats={n}, T={temp}, top_p={top_p}, max_gen_toks={max_toks})")
PY

echo ""
echo "==============================================================="
echo " OmniZipEval · DeepSeek-R1 GPQA (report protocol)"
echo " Model:        $MODEL"
echo " Target (ref): Llama-8B Diamond≈49.0 | Qwen-7B Diamond≈49.1"
echo " GPUs:         $GPUS  (vLLM tensor_parallel_size=$TP_SIZE)"
echo " Backend:      $BACKEND"
echo " N_SAMPLES:    $N_SAMPLES   (report=64)"
echo " Decode:       T=$TEMPERATURE top_p=$TOP_P max_gen_toks=$MAX_GEN_TOKS"
echo " Few-shot:     $NUM_FEWSHOT (forced)"
echo " Out:          $BASE_OUT"
echo "==============================================================="
if [[ "$N_SAMPLES" != "64" ]]; then
  echo "[warn] N_SAMPLES=$N_SAMPLES ≠ 64 → not fully report-faithful (Average Pass@1 will differ)."
fi

build_model_args() {
  if [[ "$BACKEND" == "vllm" ]]; then
    if ! python -c "import vllm" >/dev/null 2>&1; then
      die "BACKEND=vllm but vllm is not installed. pip install vllm  (or BACKEND=hf)"
    fi
    # vLLM: TP only (no PP flag in lm-eval vllm wrapper for most versions).
    local tp="$NUM_GPUS"
    echo "pretrained=${MODEL},dtype=${DTYPE},tensor_parallel_size=${tp},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},trust_remote_code=${TRUST_REMOTE_CODE}"
  else
    echo "pretrained=${MODEL},dtype=${DTYPE},parallelize=True,trust_remote_code=${TRUST_REMOTE_CODE}"
  fi
}

run_task() {
  local task="$1"
  local split_name="$2"
  local out="${BASE_OUT}/${split_name}_n${N_SAMPLES}"
  ensure_dir "$out"

  local model_args
  model_args="$(build_model_args)"

  local extra=()
  if [[ -n "$LIMIT" ]]; then
    extra+=(--limit "$LIMIT")
  fi

  echo ""
  echo "[run] task=$task  split=$split_name  → $out"

  # No --system_instruction (DeepSeek: put all instructions in the user prompt).
  python -m lm_eval \
    --model "$BACKEND" \
    --model_args "$model_args" \
    --tasks "$task" \
    --include_path "$TASK_RUNTIME" \
    --num_fewshot "$NUM_FEWSHOT" \
    --apply_chat_template \
    --gen_kwargs "temperature=${TEMPERATURE},top_p=${TOP_P},max_gen_toks=${MAX_GEN_TOKS},do_sample=True" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$out" \
    --log_samples \
    "${extra[@]}"

  echo "[done] $split_name → $out"
  echo "       Primary metric to compare with the report:  pass_at_1  (Average Pass@1)"
}

case "$ONLY" in
  both)
    run_task gpqa_main_deepseek_r1_cot main
    run_task gpqa_diamond_deepseek_r1_cot diamond
    ;;
  main)
    run_task gpqa_main_deepseek_r1_cot main
    ;;
  diamond)
    run_task gpqa_diamond_deepseek_r1_cot diamond
    ;;
  *)
    die "ONLY must be both|main|diamond (got: $ONLY)"
    ;;
esac

cat <<EOF

Finished.
Results: $BASE_OUT
Compare diamond pass_at_1 to the technical report:
  deepseek-ai/DeepSeek-R1-Distill-Llama-8B → ~49.0
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  → ~49.1
Note: full N=64 × 198 Diamond questions with long CoT is expensive; variance remains even at N=64.
EOF
