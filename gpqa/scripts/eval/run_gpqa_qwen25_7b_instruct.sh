#!/usr/bin/env bash
# GPQA for Qwen/Qwen2.5-7B-Instruct matching the Qwen2.5 technical report.
#
# Report target (arXiv:2412.15115, Instruct tables / blog):
#   Qwen2.5-7B-Instruct · GPQA ≈ 36.4  (GPQA Diamond)
#
# Protocol (Instruct models — Qwen team clarification + OpenCompass-style gen):
#   - 0-shot generative CoT  (base models in the paper use 5-shot; Instruct uses 0-shot)
#   - greedy decode: temperature=0, do_sample=False
#   - N_SAMPLES=1
#   - apply_chat_template
#   - lm-eval tasks: gpqa_*_qwen25_cot_zeroshot
#     (same cot_zeroshot prompt + fixed answer extract for boxed / Answer: / (A-D))
#   - default ONLY=both → gpqa_main + gpqa_diamond
#
# Usage:
#   bash scripts/eval/run_gpqa_qwen25_7b_instruct.sh
#   GPUS=0,1 bash scripts/eval/run_gpqa_qwen25_7b_instruct.sh
#   ONLY=diamond LIMIT=16 bash scripts/eval/run_gpqa_qwen25_7b_instruct.sh
#
# Requires: pip install lm-eval   (+ vllm recommended)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Qwen2.5-7B has 28 attention heads → TP must divide 28 (1, 2, 4, 7). Not 8.
GPUS="${GPUS:-0,1,2,3}"
resolve_gpus
check_lm_eval

case "$NUM_GPUS" in
  1|2|4) ;;
  *)
    die "NUM_GPUS=$NUM_GPUS is not a valid vLLM TP size for Qwen2.5-7B (28 heads).
Use 1, 2, or 4 GPUs. Example: GPUS=0,1,2,3 bash $0"
    ;;
esac
TP_SIZE="$NUM_GPUS"

# ---- Qwen2.5 Instruct / report defaults ------------------------------------
N_SAMPLES="${N_SAMPLES:-1}"           # forced fast / report-style single sample
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NUM_FEWSHOT=0                         # Instruct GPQA = 0-shot (Qwen #931)
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BACKEND="${BACKEND:-vllm}"            # vllm | hf
ONLY="${ONLY:-both}"                  # both | main | diamond
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
TASK_INCLUDE="${TASK_INCLUDE:-$SCRIPT_DIR/tasks/gpqa_qwen25}"

if [[ "$N_SAMPLES" != "1" ]]; then
  echo "[warn] Overriding N_SAMPLES=$N_SAMPLES → 1 (Qwen2.5 report uses a single greedy sample)." >&2
  N_SAMPLES=1
fi

TAG="$(safe_name "$(model_tag "$MODEL")")"
BASE_OUT="${OUT_DIR:-$EVAL_OUTPUT_ROOT/gpqa_qwen25/${TAG}}"
ensure_dir "$BASE_OUT"

echo ""
echo "==============================================================="
echo " OmniZipEval · Qwen2.5 Instruct GPQA (report protocol)"
echo " Model:        $MODEL"
echo " Target (ref): Qwen2.5-7B-Instruct GPQA ≈ 36.4 (Diamond)"
echo " GPUs:         $GPUS  (vLLM tensor_parallel_size=$TP_SIZE)"
echo " Backend:      $BACKEND"
echo " N_SAMPLES:    $N_SAMPLES"
echo " Decode:       greedy T=$TEMPERATURE max_gen_toks=$MAX_GEN_TOKS"
echo " Few-shot:     $NUM_FEWSHOT (Instruct = 0-shot)"
echo " Splits:       $ONLY"
echo " Tasks:        $TASK_INCLUDE"
echo " Out:          $BASE_OUT"
echo "==============================================================="

build_model_args() {
  if [[ "$BACKEND" == "vllm" ]]; then
    if ! python -c "import vllm" >/dev/null 2>&1; then
      die "BACKEND=vllm but vllm is not installed. pip install vllm  (or BACKEND=hf)"
    fi
    echo "pretrained=${MODEL},dtype=${DTYPE},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},trust_remote_code=${TRUST_REMOTE_CODE}"
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

  python -m lm_eval \
    --model "$BACKEND" \
    --model_args "$model_args" \
    --tasks "$task" \
    --include_path "$TASK_INCLUDE" \
    --num_fewshot "$NUM_FEWSHOT" \
    --apply_chat_template \
    --gen_kwargs "temperature=${TEMPERATURE},top_p=${TOP_P},max_gen_toks=${MAX_GEN_TOKS},do_sample=False" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$out" \
    --log_samples \
    "${extra[@]}"

  echo "[done] $split_name → $out"
  echo "       Primary metric: exact_match,strict-match  (report Diamond ≈ 36.4)"
}

case "$ONLY" in
  both)
    run_task gpqa_main_qwen25_cot_zeroshot main
    run_task gpqa_diamond_qwen25_cot_zeroshot diamond
    ;;
  main)
    run_task gpqa_main_qwen25_cot_zeroshot main
    ;;
  diamond)
    run_task gpqa_diamond_qwen25_cot_zeroshot diamond
    ;;
  *)
    die "ONLY must be both|main|diamond (got: $ONLY)"
    ;;
esac

cat <<EOF

Finished.
Results: $BASE_OUT
Compare diamond exact_match,strict-match to Qwen2.5 technical report:
  Qwen/Qwen2.5-7B-Instruct → GPQA ≈ 36.4
EOF
