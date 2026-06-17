# Reproducing the Submitted Runs

This package contains the five Kaggle submission files and the source code used
to generate them. The scripts in `src/runs/` are wrappers around the shared code
in `src/common/`.

## Contents

- `submissions/submission.csv`: lexical BM25 + RRF baseline, public score 0.43145.
- `submissions/submission_ltr_v2_k1000.csv`: LTR v2 with `candidate_k=1000`.
- `submissions/submission_ltr_v3_dense.csv`: V3 dense + ensemble, public score 0.50457.
- `submissions/submission_ltr_v3_cross.csv`: V3 dense + cross-encoder + ensemble, public score 0.50746.
- `submissions/submission_ltr_v3_qwen4b.csv`: V3 dense + Qwen/Qwen3-Reranker-4B + ensemble, public score 0.53561.

## Prerequisites

Run commands from the repository root. The scripts assume the original Kaggle
input files are available under `data/`, and that the SQLite FTS index exists at
`artifacts/entities.sqlite`. If the index is missing, rebuild it with:

```bash
python package/src/common/build_index_v2.py
```

Install dependencies with:

```bash
pip install -r package/src/requirements.txt
```

The Qwen reranker run is computationally expensive and should be run on a GPU.

## Dry Run

To verify paths and commands without training models or overwriting CSVs:

```bash
DRY_RUN=1 bash package/src/runs/generate_baseline.sh
DRY_RUN=1 bash package/src/runs/generate_ltr_v2_k1000.sh
DRY_RUN=1 bash package/src/runs/generate_ltr_v3_dense.sh
DRY_RUN=1 bash package/src/runs/generate_ltr_v3_cross.sh
DRY_RUN=1 bash package/src/runs/generate_ltr_v3_qwen4b.sh
```

You can choose a Python executable with `PYTHON_BIN`, for example:

```bash
PYTHON_BIN=.rc-venv/bin/python DRY_RUN=1 bash package/src/runs/generate_ltr_v3_dense.sh
```

## Generate Submissions

Run each script without `DRY_RUN=1` to regenerate the corresponding CSV in
`package/submissions/`:

```bash
bash package/src/runs/generate_baseline.sh
bash package/src/runs/generate_ltr_v2_k1000.sh
bash package/src/runs/generate_ltr_v3_dense.sh
bash package/src/runs/generate_ltr_v3_cross.sh
bash package/src/runs/generate_ltr_v3_qwen4b.sh
```

The LTR scripts retrain models before writing the submissions. They may take a
long time to run and may update model files under `artifacts/`.


