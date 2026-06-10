from __future__ import annotations

import csv
from pathlib import Path

from rcsearch.eval import ndcg_at_k
from rcsearch.export import export_submission
from rcsearch.retrieve import RunRow, write_run


def test_ndcg_perfect_ranking() -> None:
    qrels = {"001": {"A": 2, "B": 1}}
    rows = [RunRow("001", "A", 1, 1.0), RunRow("001", "B", 2, 0.5)]
    assert ndcg_at_k(rows, qrels, k=100) == 1.0


def test_ndcg_empty_ranking() -> None:
    qrels = {"001": {"A": 2}}
    assert ndcg_at_k([], qrels, k=100) == 0.0


def test_export_submission_validates_and_deduplicates(tmp_path: Path) -> None:
    queries = tmp_path / "test_queries.csv"
    queries.write_text("QueryId,Query\n001,test query\n", encoding="utf-8")
    run = tmp_path / "run.csv"
    write_run(run, [RunRow("001", "0000001", 1, 1.0), RunRow("001", "0000001", 2, 0.9)])
    out = tmp_path / "submission.csv"
    export_submission(run, out, queries, top_k=100)
    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows == [["QueryId", "EntityId"], ["001", "0000001"]]
