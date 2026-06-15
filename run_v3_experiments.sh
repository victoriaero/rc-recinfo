#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_v3_experiments.sh
#
# Optional:
#   PYTHON_BIN=.rc-venv/bin/python bash run_v3_experiments.sh
#   DEVICE=cuda bash run_v3_experiments.sh
#   CANDIDATE_K=1000 NUM_BOOST_ROUND=300 bash run_v3_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cuda}"
CANDIDATE_K="${CANDIDATE_K:-1000}"
NUM_BOOST_ROUND="${NUM_BOOST_ROUND:-300}"
CROSS_ENCODER_TOP_K="${CROSS_ENCODER_TOP_K:-200}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="artifacts/v3_runs/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
MODEL_DIR="${RUN_DIR}/models"
SUBMISSION_DIR="${RUN_DIR}/submissions"

mkdir -p "${LOG_DIR}" "${MODEL_DIR}" "${SUBMISSION_DIR}"

echo "=============================================="
echo "V3 experiments"
echo "Run dir: ${RUN_DIR}"
echo "Python: ${PYTHON_BIN}"
echo "Device: ${DEVICE}"
echo "Candidate K: ${CANDIDATE_K}"
echo "Boost rounds: ${NUM_BOOST_ROUND}"
echo "Cross-encoder top-k: ${CROSS_ENCODER_TOP_K}"
echo "=============================================="

copy_model() {
  local name="$1"
  local src="artifacts/ltr_model_v3.pkl"
  local dst="${MODEL_DIR}/ltr_model_v3_${name}.pkl"

  if [[ ! -f "${src}" ]]; then
    echo "ERROR: expected model not found at ${src}" >&2
    exit 1
  fi

  cp "${src}" "${dst}"
  echo "Saved model: ${dst}"
}

run_step() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo ""
  echo "=============================================="
  echo "Running: ${name}"
  echo "Log: ${log_file}"
  echo "=============================================="

  "$@" 2>&1 | tee "${log_file}"
}

# 1) Lexical + ensemble
run_step "train_lexical_ensemble" \
  "${PYTHON_BIN}" ltr_pipeline_v3.py \
    --mode train \
    --candidate-k "${CANDIDATE_K}" \
    --ensemble \
    --use-global-expander \
    --num-boost-round "${NUM_BOOST_ROUND}"

copy_model "lexical_ensemble"

run_step "submission_lexical_ensemble" \
  "${PYTHON_BIN}" ltr_pipeline_v3.py \
    --mode submission \
    --out "${SUBMISSION_DIR}/submission_ltr_v3_lexical_ensemble.csv"

# 2) Dense + ensemble
run_step "train_dense" \
  "${PYTHON_BIN}" ltr_pipeline_v3.py \
    --mode train \
    --candidate-k "${CANDIDATE_K}" \
    --ensemble \
    --use-global-expander \
    --use-dense \
    --device "${DEVICE}" \
    --num-boost-round "${NUM_BOOST_ROUND}"

copy_model "dense"

run_step "submission_dense" \
  "${PYTHON_BIN}" ltr_pipeline_v3.py \
    --mode submission \
    --out "${SUBMISSION_DIR}/submission_ltr_v3_dense.csv" \
    --device "${DEVICE}"

# 3) Dense + cross-encoder + ensemble
run_step "train_cross" \
  "${PYTHON_BIN}" ltr_pipeline_v3.py \
    --mode train \
    --candidate-k "${CANDIDATE_K}" \
    --ensemble \
    --use-global-expander \
    --use-dense \
    --use-cross-encoder \
    --cross-encoder-top-k "${CROSS_ENCODER_TOP_K}" \
    --device "${DEVICE}" \
    --num-boost-round "${NUM_BOOST_ROUND}"

copy_model "cross"

run_step "submission_cross" \
  "${PYTHON_BIN}" ltr_pipeline_v3.py \
    --mode submission \
    --out "${SUBMISSION_DIR}/submission_ltr_v3_cross.csv" \
    --device "${DEVICE}"

echo ""
echo "=============================================="
echo "Done."
echo "Outputs saved in:"
echo "  ${RUN_DIR}"
echo ""
echo "Submissions:"
ls -lh "${SUBMISSION_DIR}"
echo ""
echo "Models:"
ls -lh "${MODEL_DIR}"
echo ""
echo "Logs:"
ls -lh "${LOG_DIR}"
echo "=============================================="
