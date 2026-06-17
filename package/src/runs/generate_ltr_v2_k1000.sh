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

# Train the lexical LambdaMART model with a 1000-candidate pool.
run_cmd "${PYTHON_BIN}" package/src/common/ltr_pipeline_v2.py \
  --mode train \
  --candidate-k 1000 \
  --top-k 100 \
  --rrf-k 60 \
  --num-boost-round 260

# Reuse the saved model to write the Kaggle submission file.
run_cmd "${PYTHON_BIN}" package/src/common/ltr_pipeline_v2.py \
  --mode submission \
  --out package/submissions/submission_ltr_v2_k1000.csv
