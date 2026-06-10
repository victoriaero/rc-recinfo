from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

from .data import read_qrels
from .retrieve import RunRow, read_run


def dcg(relevances: Iterable[int], k: int = 100) -> float:
    total = 0.0
    for idx, rel in enumerate(list(relevances)[:k], start=1):
        total += (2**rel - 1) / math.log2(idx + 1)
    return total


def ndcg_at_k(run_rows: list[RunRow], qrels: dict[str, dict[str, int]], k: int = 100) -> float:
    by_query: dict[str, list[RunRow]] = {}
    for row in run_rows:
        by_query.setdefault(row.query_id, []).append(row)
    scores = []
    for query_id, rels in qrels.items():
        ranking = sorted(by_query.get(query_id, []), key=lambda row: row.rank)[:k]
        gains = [rels.get(row.entity_id, 0) for row in ranking]
        ideal = sorted(rels.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal, k)
        scores.append(0.0 if ideal_dcg == 0 else dcg(gains, k) / ideal_dcg)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_run(run_path: str | Path, qrels_path: str | Path, k: int = 100) -> float:
    return ndcg_at_k(read_run(run_path), read_qrels(qrels_path), k=k)


def write_grid_results(path: str | Path, rows: list[dict[str, float]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["title", "keywords", "text", "ndcg_at_100"])
        writer.writeheader()
        writer.writerows(rows)
