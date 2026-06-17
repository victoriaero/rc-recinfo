from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

# Build the original compact SQLite/FTS5 index used by the baseline pipeline.
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "entities.sqlite"


def iter_corpus(corpus_path: Path, limit: int | None = None):
    with corpus_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            yield (
                str(obj["id"]),
                str(obj.get("title") or ""),
                " ".join(obj.get("keywords") or []),
                str(obj.get("text") or ""),
            )


def connect() -> sqlite3.Connection:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    con = sqlite3.connect(INDEX_PATH)
    con.execute("pragma journal_mode=WAL")
    con.execute("pragma synchronous=NORMAL")
    con.execute("pragma temp_store=MEMORY")
    con.execute("pragma cache_size=-2000000")
    return con


def create_schema(con: sqlite3.Connection) -> None:
    for table in ["docs", "docs_porter", "entities"]:
        con.execute(f"drop table if exists {table}")

    # Keep normalized fields in a regular table for feature extraction.
    con.execute(
        """
        create table entities (
            entity_id text primary key,
            title text not null,
            keywords text not null,
            text text not null
        )
        """
    )

    # Create both raw and Porter-stemmed FTS views over the same fields.
    for table, tokenizer in [
        ("docs", "unicode61 remove_diacritics 2"),
        ("docs_porter", "porter unicode61 remove_diacritics 2"),
    ]:
        con.execute(
            f"""
            create virtual table {table} using fts5(
                entity_id unindexed,
                title,
                keywords,
                text,
                tokenize='{tokenizer}'
            )
            """
        )
    con.commit()


def insert_batch(con: sqlite3.Connection, batch: list[tuple[str, str, str, str]]) -> None:
    con.executemany("insert into entities(entity_id, title, keywords, text) values (?, ?, ?, ?)", batch)
    for table in ["docs", "docs_porter"]:
        con.executemany(f"insert into {table}(entity_id, title, keywords, text) values (?, ?, ?, ?)", batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Index only the first N entities; useful for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    con = connect()
    create_schema(con)

    batch = []
    total = 0
    with con:
        for row in tqdm(iter_corpus(DATA_DIR / "corpus.jsonl", args.limit), desc="indexing", unit="entity"):
            batch.append(row)
            if len(batch) >= args.batch_size:
                insert_batch(con, batch)
                total += len(batch)
                batch.clear()
        if batch:
            insert_batch(con, batch)
            total += len(batch)
    con.close()
    print(f"Indexed {total} entities into {INDEX_PATH}")


if __name__ == "__main__":
    main()
