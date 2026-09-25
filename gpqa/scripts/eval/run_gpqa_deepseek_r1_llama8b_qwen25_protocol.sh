#!/usr/bin/env bash
# GPQA for deepseek-ai/DeepSeek-R1-Distill-Llama-8B using the *Qwen2.5 Instruct*
# evaluation protocol (for fair comparison with run_gpqa_qwen25_7b_instruct.sh).
#
# This is NOT the DeepSeek-R1 report protocol (that is run_gpqa_deepseek_r1_llama8b.sh:
# T=0.6, top_p=0.95, boxed + <think>, N up to 64).
#
# Shared Qwen2.5-style protocol:
#   - dataset: Idavidrein/gpqa  (main + diamond by default)
#   - task: gpqa_*_qwen25_cot_zeroshot  (same prompt as cot_zeroshot + fixed answer parse)
#   - 0-shot generative CoT ("Let's think step by step:")
#   - greedy: temperature=0, do_sample=False
#   - N_SAMPLES=1
#   - apply_chat_template
#   - max_gen_toks=8192
#   - strict-match: final answer from boxed / Answer: / The answer is / (A-D)
#
# Usage:
#   bash scripts/eval/run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh
#   GPUS=0,1 bash scripts/eval/run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh
#   ONLY=diamond LIMIT=16 bash scripts/eval/run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh
#
# Requires: pip install lm-eval   (+ vllm recommended)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
# Llama-3.1-8B: 32 heads → TP in {1,2,4,8}
GPUS="${GPUS:-0,1,2,3}"
resolve_gpus
check_lm_eval

case "$NUM_GPUS" in
  1|2|4|8) ;;
  *)
    die "NUM_GPUS=$NUM_GPUS is not a valid vLLM TP size for Llama-8B (32 heads).
Use 1, 2, 4, or 8 GPUs. Example: GPUS=0,1,2,3 bash $0"
    ;;
esac
TP_SIZE="$NUM_GPUS"

# ---- identical to run_gpqa_qwen25_7b_instruct.sh ----------------------------
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NUM_FEWSHOT=0
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BACKEND="${BACKEND:-vllm}"
ONLY="${ONLY:-both}"                 # both | main | diamond
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
TASK_INCLUDE="${TASK_INCLUDE:-$SCRIPT_DIR/tasks/gpqa_qwen25}"

if [[ "$N_SAMPLES" != "1" ]]; then
  echo "[warn] Overriding N_SAMPLES=$N_SAMPLES → 1 (Qwen2.5 protocol = single greedy sample)." >&2
  N_SAMPLES=1
fi

TAG="$(safe_name "$(model_tag "$MODEL")")"
BASE_OUT="${OUT_DIR:-$EVAL_OUTPUT_ROOT/gpqa_qwen25_protocol/${TAG}}"
ensure_dir "$BASE_OUT"

echo ""
echo "==============================================================="
echo " OmniZipEval · DeepSeek-R1 Llama-8B under Qwen2.5 GPQA protocol"
echo " Model:        $MODEL"
echo " Protocol:     same as run_gpqa_qwen25_7b_instruct.sh"
echo "               (0-shot, greedy, N=1) + fixed answer extract"
echo " Splits:       $ONLY  (main=gpqa_main, diamond=gpqa_diamond)"
echo " Tasks:        $TASK_INCLUDE"
echo " GPUs:         $GPUS  (vLLM tensor_parallel_size=$TP_SIZE)"
echo " Backend:      $BACKEND"
echo " N_SAMPLES:    $N_SAMPLES"
echo " Decode:       greedy T=$TEMPERATURE max_gen_toks=$MAX_GEN_TOKS"
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
  echo "       Primary metric: exact_match,strict-match  (fixed final-answer extract)"
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
Report exact_match,strict-match (and flexible-extract as secondary).
EOF
