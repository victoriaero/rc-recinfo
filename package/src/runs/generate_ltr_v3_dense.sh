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

# Train the V3 ensemble with global expansion and dense semantic features.
run_cmd "${PYTHON_BIN}" package/src/common/ltr_pipeline_v3.py \
  --mode train \
  --candidate-k 1000 \
  --ensemble \
  --use-global-expander \
  --use-dense \
  --num-boost-round 300

# Reuse the saved dense V3 model to write the submission file.
run_cmd "${PYTHON_BIN}" package/src/common/ltr_pipeline_v3.py \
  --mode submission \
  --out package/submissions/submission_ltr_v3_dense.csv
