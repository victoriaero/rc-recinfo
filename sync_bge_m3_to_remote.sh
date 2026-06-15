#!/usr/bin/env bash
set -euo pipefail

REMOTE="samiramalaquias@192.168.62.39"
DEST="~/rc-recinfo"
SSH_OPTS=(
  -o "ProxyCommand=ssh -q -W %h:%p cerberus"
  -o "IdentitiesOnly=yes"
)

FILES=(
  "ltr_pipeline_v3.py"
  "ltr_pipeline_v3_bge_m3.py"
  "requirements.txt"
  "requirements_bge_m3.txt"
  "data/train_queries.csv"
  "data/train_qrels.csv"
  "data/test_queries.csv"
  "artifacts/entities.sqlite"
)

ssh "${SSH_OPTS[@]}" "${REMOTE}" "mkdir -p ${DEST}/data ${DEST}/artifacts"

rsync -avhP --relative \
  -e "ssh -o ProxyCommand='ssh -q -W %h:%p cerberus' -o IdentitiesOnly=yes" \
  "${FILES[@]}" \
  "${REMOTE}:${DEST}/"
