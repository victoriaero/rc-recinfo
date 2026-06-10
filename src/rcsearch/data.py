from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Entity:
    entity_id: str
    title: str
    keywords: list[str]
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def read_queries(path: str | Path) -> list[Query]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [Query(row["QueryId"], row["Query"]) for row in reader]


def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            qrels.setdefault(row["QueryId"], {})[row["EntityId"]] = int(row["Relevance"])
    return qrels


def iter_corpus(path: str | Path, limit: int | None = None) -> Iterator[Entity]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            yield Entity(
                entity_id=str(obj["id"]),
                title=str(obj.get("title") or ""),
                keywords=list(obj.get("keywords") or []),
                text=str(obj.get("text") or ""),
            )


def batched(items: Iterable[Entity], batch_size: int) -> Iterator[list[Entity]]:
    batch: list[Entity] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
