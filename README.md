# RC Entity Search Pipeline

Pipeline progressivo para o Research Challenge de Entity Search usando SQLite FTS5 como motor lexical principal.

## Uso básico

```bash
PYTHONPATH=src .rc-venv/bin/python -m rcsearch build-index --config configs/submission_01.yaml --reset
PYTHONPATH=src .rc-venv/bin/python -m rcsearch retrieve --config configs/submission_01.yaml --split test
PYTHONPATH=src .rc-venv/bin/python -m rcsearch export --run artifacts/runs/submission_01_test.csv --out artifacts/submissions/submission_01.csv
```

Para avaliar no treino:

```bash
PYTHONPATH=src .rc-venv/bin/python -m rcsearch retrieve --config configs/submission_01.yaml --split train
PYTHONPATH=src .rc-venv/bin/python -m rcsearch evaluate --run artifacts/runs/submission_01_train.csv
```

Para buscar pesos fielded:

```bash
PYTHONPATH=src .rc-venv/bin/python -m rcsearch grid-search --config configs/submission_04.yaml
```

A submissao 05 espera candidatos em `artifacts/runs/submission_05_train_candidates.csv` e `artifacts/runs/submission_05_test_candidates.csv`, gerados com `retrieve --config configs/submission_05.yaml --split train` e `--split test`.
