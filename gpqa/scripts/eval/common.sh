#!/usr/bin/env bash
# Shared helpers for OmniZipEval benchmark launchers.
# Sourced by scripts under scripts/eval/ — do not run directly.
#
# Convention:
#   - OmniZipEval stays thin; heavy suites live in sibling clones.
#   - Override roots with LONGBENCH_ROOT / LCB_ROOT / VLMEVAL_ROOT.
#   - Default sibling layout: <parent>/{OmniZipEval,LongBench,LiveCodeBench,VLMEvalKit}

set -euo pipefail

OMNIZIP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT_ROOT="$(cd "$OMNIZIP_ROOT/.." && pwd)"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$OMNIZIP_ROOT/outputs}"

die() {
  echo "Error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"
}

safe_name() {
  echo "$1" | tr '/:' '__' | tr -c 'A-Za-z0-9._-' '_'
}

model_tag() {
  # Last path component for HF ids or local dirs.
  local m="$1"
  basename "${m%/}"
}

resolve_gpus() {
  # Prefer explicit GPUS=; else CUDA_VISIBLE_DEVICES; else 0.
  if [[ -n "${GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPUS="$CUDA_VISIBLE_DEVICES"
  else
    GPUS="${GPUS:-0}"
    export CUDA_VISIBLE_DEVICES="$GPUS"
  fi
  IFS=',' read -r -a GPU_ARR <<< "$GPUS"
  NUM_GPUS="${#GPU_ARR[@]}"
  (( NUM_GPUS >= 1 )) || die "no GPUs in GPUS=\"$GPUS\""
}

infer_tp_pp() {
  # Pick TP that divides NUM_GPUS (prefer 4, then 2, then 1); PP fills the rest.
  if [[ -n "${TP_SIZE:-}" ]]; then
    (( NUM_GPUS % TP_SIZE == 0 )) || die "TP_SIZE=$TP_SIZE does not divide NUM_GPUS=$NUM_GPUS"
  elif (( NUM_GPUS % 4 == 0 )); then
    TP_SIZE=4
  elif (( NUM_GPUS % 2 == 0 )); then
    TP_SIZE=2
  else
    TP_SIZE=1
  fi
  PP_SIZE=$(( NUM_GPUS / TP_SIZE ))
}

require_model() {
  [[ -n "${MODEL:-}" ]] || die "MODEL is required (HF id or local checkpoint path)

Example:
  MODEL=Qwen/Qwen2.5-7B-Instruct GPUS=0,1 bash $0"
}

ensure_dir() {
  mkdir -p "$1"
}

print_banner() {
  local title="$1"
  echo "==============================================================="
  echo " OmniZipEval · $title"
  echo " Model:  ${MODEL:-<unset>}"
  echo " GPUs:   ${GPUS:-<unset>}  (n=${NUM_GPUS:-?})"
  echo " Out:    ${OUT_DIR:-<unset>}"
  echo "==============================================================="
}

require_external_root() {
  # usage: require_external_root VAR_NAME default_path clone_url setup_hint
  local var_name="$1"
  local default_path="$2"
  local clone_url="$3"
  local hint="$4"
  local root="${!var_name:-$default_path}"

  if [[ ! -d "$root" ]]; then
    cat >&2 <<EOF
Error: ${var_name} not found at:
  $root

Clone the upstream suite next to OmniZipEval (recommended):

  bash scripts/setup_benchmarks.sh
  # or manually:
  git clone ${clone_url} ${default_path}

Then install deps inside that repo (${hint}) and re-run.

Override path with:
  export ${var_name}=/path/to/repo
EOF
    exit 1
  fi
  printf -v "$var_name" '%s' "$root"
  export "$var_name"
}

check_lm_eval() {
  if ! python -c "import lm_eval" >/dev/null 2>&1; then
    die "lm-eval is not installed in the active Python env.

  pip install lm-eval
  # or: pip install 'lm-eval[hf]'

See docs/EVAL.md § LLM (lm-evaluation-harness)."
  fi
}
