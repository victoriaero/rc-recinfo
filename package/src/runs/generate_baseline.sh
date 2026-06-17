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

# Generate the lexical BM25/RRF baseline submission.
run_cmd "${PYTHON_BIN}" package/src/common/make_submission.py \
  --out package/submissions/submission.csv
