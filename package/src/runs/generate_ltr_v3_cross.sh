#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

# Print every command and skip execution when DRY_RUN=1.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

cd "${REPO_ROOT}"

# Train V3 with dense features plus a top-200 cross-encoder feature block.
run_cmd "${PYTHON_BIN}" package/src/common/ltr_pipeline_v3.py \
  --mode train \
  --candidate-k 1000 \
  --ensemble \
  --use-global-expander \
  --use-dense \
  --use-cross-encoder \
  --cross-encoder-top-k 200 \
  --num-boost-round 300

# Reuse the saved cross-encoder V3 model to write the submission file.
run_cmd "${PYTHON_BIN}" package/src/common/ltr_pipeline_v3.py \
  --mode submission \
  --out package/submissions/submission_ltr_v3_cross.csv
