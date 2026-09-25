#!/usr/bin/env bash
# GPQA for DeepSeek-R1 Distill W8A8-SmoothQuant models under the *Qwen2.5 Instruct*
# protocol (same as run_gpqa_qwen25_7b_instruct.sh /
# run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh).
#
# Models (run sequentially by default):
#   1. felixnguyen33/DeepSeek-R1-Distill-Qwen-7B-W8A8-SmoothQuant
#   2. felixnguyen33/DeepSeek-R1-Distill-Llama-8B-W8A8-SmoothQuant
#
# Protocol:
#   - dataset: Idavidrein/gpqa  (main + diamond by default)
#   - task: gpqa_*_qwen25_cot_zeroshot  (fixed answer extract)
#   - 0-shot generative CoT, greedy T=0, N_SAMPLES=1
#   - apply_chat_template, max_gen_toks=8192
#   - vLLM: quantization=compressed-tensors
#
# Usage:
#   bash scripts/eval/run_gpqa_deepseek_r1_smoothquant_qwen25_protocol.sh
#   GPUS=0,1,2,3 bash scripts/eval/run_gpqa_deepseek_r1_smoothquant_qwen25_protocol.sh
#   ONLY_MODEL=qwen   bash scripts/eval/run_gpqa_deepseek_r1_smoothquant_qwen25_protocol.sh
#   ONLY_MODEL=llama  bash scripts/eval/run_gpqa_deepseek_r1_smoothquant_qwen25_protocol.sh
#   ONLY=diamond LIMIT=16 bash scripts/eval/run_gpqa_deepseek_r1_smoothquant_qwen25_protocol.sh
#
# Requires: pip install lm-eval vllm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

GPUS="${GPUS:-0,1,2,3}"
resolve_gpus
check_lm_eval

# TP=4 works for both Qwen-7B (28 heads) and Llama-8B (32 heads).
case "$NUM_GPUS" in
  1|2|4) ;;
  *)
    die "NUM_GPUS=$NUM_GPUS must be 1, 2, or 4 for this SmoothQuant pair
(Qwen-7B has 28 heads — TP=8 is invalid). Example: GPUS=0,1,2,3 bash $0"
    ;;
esac
TP_SIZE="$NUM_GPUS"

# ---- identical decode protocol to Qwen2.5 Instruct GPQA script -------------
N_SAMPLES="${N_SAMPLES:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
NUM_FEWSHOT=0
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
BACKEND="${BACKEND:-vllm}"
QUANTIZATION="${QUANTIZATION:-compressed-tensors}"
ONLY="${ONLY:-both}"             # both | main | diamond  (GPQA split)
ONLY_MODEL="${ONLY_MODEL:-both}" # both | qwen | llama
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
TASK_INCLUDE="${TASK_INCLUDE:-$SCRIPT_DIR/tasks/gpqa_qwen25}"

if [[ "$N_SAMPLES" != "1" ]]; then
  echo "[warn] Overriding N_SAMPLES=$N_SAMPLES → 1 (Qwen2.5 protocol = single greedy sample)." >&2
  N_SAMPLES=1
fi

# model_id | short_tag | tokenizer_override (empty = use checkpoint tokenizer)
MODELS=(
  "felixnguyen33/DeepSeek-R1-Distill-Qwen-7B-W8A8-SmoothQuant|DeepSeek-R1-Distill-Qwen-7B-W8A8-SmoothQuant|"
  "felixnguyen33/DeepSeek-R1-Distill-Llama-8B-W8A8-SmoothQuant|DeepSeek-R1-Distill-Llama-8B-W8A8-SmoothQuant|deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

should_run_model() {
  local tag="$1"
  case "$ONLY_MODEL" in
    both) return 0 ;;
    qwen|Qwen|Qwen7B)
      [[ "$tag" == *Qwen* ]] && return 0
      return 1
      ;;
    llama|Llama|Llama8B)
      [[ "$tag" == *Llama* ]] && return 0
      return 1
      ;;
    *)
      # substring / exact HF id match
      [[ "$tag" == *"$ONLY_MODEL"* || "$1" == *"$ONLY_MODEL"* ]] && return 0
      return 1
      ;;
  esac
}

if [[ "$BACKEND" == "vllm" ]] && ! python -c "import vllm" >/dev/null 2>&1; then
  die "BACKEND=vllm but vllm is not installed. pip install vllm  (or BACKEND=hf)"
fi

echo ""
echo "==============================================================="
echo " OmniZipEval · DeepSeek-R1 SmoothQuant under Qwen2.5 GPQA protocol"
echo " Protocol:     same as run_gpqa_qwen25_7b_instruct.sh"
echo "               (0-shot cot_zeroshot, greedy, N=1)"
echo " GPUs:         $GPUS  (vLLM tensor_parallel_size=$TP_SIZE)"
echo " Backend:      $BACKEND  quant=$QUANTIZATION"
echo " N_SAMPLES:    $N_SAMPLES"
echo " Decode:       greedy T=$TEMPERATURE max_gen_toks=$MAX_GEN_TOKS"
echo " GPQA split:   $ONLY    ONLY_MODEL=$ONLY_MODEL"
echo "==============================================================="

build_model_args() {
  local model="$1"
  local tokenizer="$2"
  if [[ "$BACKEND" == "vllm" ]]; then
    local args="pretrained=${model},dtype=${DTYPE},tensor_parallel_size=${TP_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=${MAX_MODEL_LEN},trust_remote_code=${TRUST_REMOTE_CODE},quantization=${QUANTIZATION}"
    if [[ -n "$tokenizer" ]]; then
      args="${args},tokenizer=${tokenizer}"
    fi
    echo "$args"
  else
    local args="pretrained=${model},dtype=${DTYPE},parallelize=True,trust_remote_code=${TRUST_REMOTE_CODE}"
    if [[ -n "$tokenizer" ]]; then
      args="${args},tokenizer=${tokenizer}"
    fi
    echo "$args"
  fi
}

run_gpqa_for_model() {
  local model="$1"
  local tag="$2"
  local tokenizer="$3"

  local base_out="${OUT_DIR:-$EVAL_OUTPUT_ROOT/gpqa_qwen25_protocol/${tag}}"
  ensure_dir "$base_out"

  local model_args
  model_args="$(build_model_args "$model" "$tokenizer")"

  run_one_task() {
    local task="$1"
    local split_name="$2"
    local out="${base_out}/${split_name}_n${N_SAMPLES}"
    ensure_dir "$out"

    local extra=()
    if [[ -n "$LIMIT" ]]; then
      extra+=(--limit "$LIMIT")
    fi

    echo ""
    echo "[run] model=$model"
    echo "      task=$task  split=$split_name  → $out"
    if [[ -n "$tokenizer" ]]; then
      echo "      tokenizer=$tokenizer"
    fi

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

    echo "[done] $tag / $split_name → $out"
    echo "       Primary metric: exact_match,strict-match"
  }

  case "$ONLY" in
    both)
      run_one_task gpqa_main_qwen25_cot_zeroshot main
      run_one_task gpqa_diamond_qwen25_cot_zeroshot diamond
      ;;
    main)
      run_one_task gpqa_main_qwen25_cot_zeroshot main
      ;;
    diamond)
      run_one_task gpqa_diamond_qwen25_cot_zeroshot diamond
      ;;
    *)
      die "ONLY must be both|main|diamond (got: $ONLY)"
      ;;
  esac
}

RAN=0
for entry in "${MODELS[@]}"; do
  IFS='|' read -r model tag tokenizer <<< "$entry"
  if ! should_run_model "$tag"; then
    echo "[skip] $tag (ONLY_MODEL=$ONLY_MODEL)"
    continue
  fi
  run_gpqa_for_model "$model" "$tag" "$tokenizer"
  RAN=$((RAN + 1))
done

(( RAN > 0 )) || die "No models selected (ONLY_MODEL=$ONLY_MODEL)."

cat <<EOF

Finished ($RAN model(s)).
Results under: $EVAL_OUTPUT_ROOT/gpqa_qwen25_protocol/
Fair compare with dense counterparts:
  bash scripts/eval/run_gpqa_deepseek_r1_llama8b_qwen25_protocol.sh
  bash scripts/eval/run_gpqa_qwen25_7b_instruct.sh
EOF
