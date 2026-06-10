from __future__ import annotations

import csv
from pathlib import Path

from .data import read_queries
from .retrieve import read_run


def export_submission(run_path: str | Path, output_path: str | Path, test_queries_path: str | Path, top_k: int = 100) -> Path:
    test_query_ids = {query.query_id for query in read_queries(test_queries_path)}
    by_query: dict[str, list[str]] = {query_id: [] for query_id in test_query_ids}
    seen: dict[str, set[str]] = {query_id: set() for query_id in test_query_ids}

    for row in sorted(read_run(run_path), key=lambda item: (item.query_id, item.rank)):
        if row.query_id not in test_query_ids:
            raise ValueError(f"Run contains non-test query id: {row.query_id}")
        if not row.entity_id:
            raise ValueError(f"Run contains empty entity id for query {row.query_id}")
        if row.entity_id in seen[row.query_id]:
            continue
        if len(by_query[row.query_id]) < top_k:
            by_query[row.query_id].append(row.entity_id)
            seen[row.query_id].add(row.entity_id)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["QueryId", "EntityId"])
        for query_id in sorted(by_query):
            for entity_id in by_query[query_id]:
                writer.writerow([query_id, entity_id])
    return output
