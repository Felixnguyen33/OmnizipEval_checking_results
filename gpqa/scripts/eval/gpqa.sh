#!/usr/bin/env bash
# GPQA via EleutherAI lm-evaluation-harness.
#
# Paper splits: Full set + Diamond subset.
# Default task: gpqa_diamond_n_shot (override with TASK=gpqa_main_n_shot).
#
# Usage:
#   MODEL=Qwen/Qwen2.5-7B-Instruct GPUS=0,1 bash scripts/eval/gpqa.sh
#   MODEL=... TASK=gpqa_main_n_shot NUM_FEWSHOT=5 bash scripts/eval/gpqa.sh
#
# Requires: pip install lm-eval  (no external repo clone)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

require_model
resolve_gpus
check_lm_eval

TASK="${TASK:-gpqa_diamond_n_shot}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

TAG="$(safe_name "$(model_tag "$MODEL")")"
OUT_DIR="${OUT_DIR:-$EVAL_OUTPUT_ROOT/gpqa/${TAG}}"
ensure_dir "$OUT_DIR"

print_banner "GPQA (${TASK}, ${NUM_FEWSHOT}-shot)"

python -m lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL},dtype=${DTYPE},parallelize=True,trust_remote_code=${TRUST_REMOTE_CODE}" \
  --tasks "$TASK" \
  --num_fewshot "$NUM_FEWSHOT" \
  --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS},temperature=${TEMPERATURE},do_sample=False" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR"

echo "[done] Results → $OUT_DIR"
