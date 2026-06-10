from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import batched, iter_corpus
from .metadata import write_metadata

SCHEMA_VERSION = 1


def connect(index_path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(index_path))
    con.execute("pragma journal_mode=WAL")
    con.execute("pragma synchronous=NORMAL")
    con.execute("pragma temp_store=MEMORY")
    return con


def create_schema(con: sqlite3.Connection, reset: bool = False) -> None:
    if reset:
        con.execute("drop table if exists entities")
        con.execute("drop table if exists docs")
        con.execute("drop table if exists meta")
    con.execute(
        """
        create table if not exists entities (
            entity_id text primary key,
            title text not null,
            keywords text not null,
            text text not null
        )
        """
    )
    con.execute(
        """
        create virtual table if not exists docs using fts5(
            entity_id unindexed,
            title,
            keywords,
            text,
            content,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    con.execute("create table if not exists meta (key text primary key, value text not null)")
    con.execute("insert or replace into meta(key, value) values('schema_version', ?)", (str(SCHEMA_VERSION),))
    con.commit()


def index_corpus(config: dict[str, Any], reset: bool = False, limit: int | None = None) -> int:
    data_dir = Path(config["paths"]["data_dir"])
    corpus_path = data_dir / "corpus.jsonl"
    index_path = Path(config["paths"]["index_path"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    batch_size = int(config.get("index", {}).get("batch_size", 5000))

    con = connect(index_path)
    create_schema(con, reset=reset)
    count = 0
    iterator = batched(iter_corpus(corpus_path, limit=limit), batch_size)
    with con:
        for batch in tqdm(iterator, desc="indexing", unit="batch"):
            entity_rows = []
            doc_rows = []
            for entity in batch:
                keywords = " ".join(entity.keywords)
                content = " ".join(part for part in [entity.title, keywords, entity.text] if part)
                entity_rows.append((entity.entity_id, entity.title, keywords, entity.text))
                doc_rows.append((entity.entity_id, entity.title, keywords, entity.text, content))
            con.executemany(
                "insert or replace into entities(entity_id, title, keywords, text) values (?, ?, ?, ?)",
                entity_rows,
            )
            con.executemany(
                "insert into docs(entity_id, title, keywords, text, content) values (?, ?, ?, ?, ?)",
                doc_rows,
            )
            count += len(batch)
    total = total_entities(con)
    con.execute("insert or replace into meta(key, value) values('entity_count', ?)", (str(total),))
    con.commit()
    con.close()

    write_metadata(
        Path(config["paths"]["artifact_dir"]) / "metadata" / "build_index.json",
        {
            "command": "build-index",
            "config_path": config.get("_config_path"),
            "index_path": str(index_path),
            "corpus_path": str(corpus_path),
            "indexed_this_run": count,
            "entity_count": total,
            "limit": limit,
            "reset": reset,
        },
    )
    return count


def total_entities(con: sqlite3.Connection) -> int:
    return int(con.execute("select count(*) from entities").fetchone()[0])
