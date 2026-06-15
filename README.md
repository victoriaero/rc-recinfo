# RC Entity Search - V2 simples e robusta

Pipeline enxuto com SQLite FTS5:

- `build_index.py`: cria o indice lexical.
- `make_submission.py`: gera a submissao.
- `evaluate.py`: calcula nDCG em runs de treino.

## Estrategia

A V2 usa varios retrievers BM25 simples e combina os candidatos com Reciprocal Rank Fusion:

- BM25 fielded balanceado.
- BM25 com titulo mais forte.
- BM25 com titulo + keywords mais fortes.
- BM25 com keywords mais fortes.
- Versoes com stemming Porter.
- Uma versao com stemming + remocao simples de stopwords na query.

Depois disso, aplica um reranking leve com sinais de entity search:

- cobertura dos termos da query no titulo;
- cobertura em keywords;
- query/frase no titulo;
- todos os termos no titulo ou em titulo+keywords;
- bigramas da query no titulo/keywords;
- bonus para titulos curtos com boa cobertura;
- penalizacao pequena para titulos longos com baixa cobertura.

## Comandos

Construir indice completo:

```bash
.rc-venv/bin/python build_index.py
```

Gerar submissao completa com 100 entidades por query:

```bash
.rc-venv/bin/python make_submission.py
```

Saida:

```text
artifacts/submission.csv
```

Smoke test rapido:

```bash
.rc-venv/bin/python build_index.py --limit 2000
.rc-venv/bin/python make_submission.py --candidate-k 50
```

Estimativa no treino, mais leve:

```bash
.rc-venv/bin/python make_submission.py --queries data/train_queries.csv --out artifacts/train_run.csv --candidate-k 50 --top-k 50
.rc-venv/bin/python evaluate.py artifacts/train_run.csv --k 50
```
