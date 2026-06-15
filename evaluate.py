from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path("data")


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            qrels[row["QueryId"]][row["EntityId"]] = int(row["Relevance"])
    return qrels


def read_run(path: Path) -> dict[str, list[str]]:
    run: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            run[row["QueryId"]].append(row["EntityId"])
    return run


def dcg(relevances: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def ndcg_at_k(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int = 100) -> float:
    values = []
    for query_id, rels in qrels.items():
        ranking = run.get(query_id, [])[:k]
        gains = [rels.get(entity_id, 0) for entity_id in ranking]
        ideal = sorted(rels.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal, k)
        values.append(0.0 if ideal_dcg == 0 else dcg(gains, k) / ideal_dcg)
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    run = read_run(args.run)
    qrels = read_qrels(DATA_DIR / "train_qrels.csv")
    expected_queries = set(qrels)
    missing = expected_queries - set(run)
    short = {qid: len(run[qid]) for qid in expected_queries & set(run) if len(run[qid]) < args.k}
    total_predictions = sum(len(rows) for rows in run.values())

    print(f"queries in qrels: {len(expected_queries)}")
    print(f"queries in run: {len(run)}")
    print(f"predictions in run: {total_predictions}")
    if missing or short:
        print(f"WARNING: incomplete run for nDCG@{args.k}: {len(missing)} missing queries, {len(short)} queries with fewer than {args.k} docs")

    score = ndcg_at_k(run, qrels, args.k)
    print(f"nDCG@{args.k}: {score:.6f}")


if __name__ == "__main__":
    main()
