from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .data import Query, read_queries
from .index import connect
from .metadata import write_metadata

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class RunRow:
    query_id: str
    entity_id: str
    rank: int
    score: float


def fts_query(text: str, operator: str = "OR") -> str:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return '""'
    joiner = f" {operator} "
    return joiner.join(f'"{token}"' for token in tokens)


def retrieve(config: dict[str, Any], split: str = "test", top_k: int | None = None, output: str | None = None) -> Path:
    data_dir = Path(config["paths"]["data_dir"])
    queries_path = data_dir / ("train_queries.csv" if split == "train" else "test_queries.csv")
    queries = read_queries(queries_path)
    top_k = top_k or int(config.get("retrieval", {}).get("top_k", 100))
    run_name = config.get("run", {}).get("name", "run")
    suffix = config.get("retrieval", {}).get("output_suffix")
    run_suffix = f"{split}_{suffix}" if suffix else split
    output_path = Path(output) if output else Path(config["paths"]["artifact_dir"]) / "runs" / f"{run_name}_{run_suffix}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = retrieve_rows(config, queries, top_k=top_k)
    write_run(output_path, rows)
    write_metadata(
        Path(config["paths"]["artifact_dir"]) / "metadata" / f"{run_name}_{run_suffix}_retrieve.json",
        {
            "command": "retrieve",
            "config_path": config.get("_config_path"),
            "split": split,
            "queries_path": str(queries_path),
            "run_path": str(output_path),
            "query_count": len(queries),
            "top_k": top_k,
            "weights": config.get("retrieval", {}).get("weights", {}),
            "mode": config.get("retrieval", {}).get("mode", "fielded"),
        },
    )
    return output_path


def retrieve_rows(config: dict[str, Any], queries: list[Query], top_k: int) -> list[RunRow]:
    con = connect(config["paths"]["index_path"])
    rows: list[RunRow] = []
    mode = config.get("retrieval", {}).get("mode", "fielded")
    weights = weights_for_mode(config)
    operator = config.get("retrieval", {}).get("operator", "OR")
    sql = """
        select entity_id, bm25(docs, ?, ?, ?, ?, ?) as score
        from docs
        where docs match ?
        order by score asc
        limit ?
    """
    for query in tqdm(queries, desc=f"retrieving {mode}", unit="query"):
        query_text = fts_query(query.text, operator=operator)
        try:
            hits = con.execute(sql, (*weights, query_text, top_k)).fetchall()
        except sqlite3.OperationalError:
            hits = []
        for rank, (entity_id, score) in enumerate(hits, start=1):
            rows.append(RunRow(query.query_id, entity_id, rank, float(score)))
    con.close()
    return rows


def weights_for_mode(config: dict[str, Any]) -> tuple[float, float, float, float, float]:
    retrieval = config.get("retrieval", {})
    mode = retrieval.get("mode", "fielded")
    weights = retrieval.get("weights", {})
    if mode == "concat":
        return (0.0, 0.0, 0.0, 0.0, float(weights.get("content", 1.0)))
    return (
        0.0,
        float(weights.get("title", 1.0)),
        float(weights.get("keywords", 1.0)),
        float(weights.get("text", 1.0)),
        0.0,
    )


def write_run(path: str | Path, rows: Iterable[RunRow]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["QueryId", "EntityId", "Rank", "Score"])
        for row in rows:
            writer.writerow([row.query_id, row.entity_id, row.rank, f"{row.score:.8f}"])


def read_run(path: str | Path) -> list[RunRow]:
    rows: list[RunRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(RunRow(row["QueryId"], row["EntityId"], int(row["Rank"]), float(row["Score"])))
    return rows
