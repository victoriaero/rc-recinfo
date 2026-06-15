from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "entities.sqlite"


def iter_corpus(corpus_path: Path, limit: int | None = None):
    with corpus_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break

            obj = json.loads(line)
            title = str(obj.get("title") or "")
            keywords = " ".join(obj.get("keywords") or [])
            text = str(obj.get("text") or "")
            title_keywords = f"{title} {keywords}".strip()
            all_content = f"{title} {keywords} {text}".strip()

            yield (
                str(obj["id"]),
                title,
                keywords,
                text,
                title_keywords,
                all_content,
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
    tables = [
        "docs",
        "docs_porter",
        "docs_all",
        "docs_title",
        "docs_keywords",
        "docs_title_keywords",
        "docs_text",
        "docs_all_porter",
        "docs_title_porter",
        "docs_keywords_porter",
        "docs_title_keywords_porter",
        "entities",
    ]

    for table in tables:
        con.execute(f"drop table if exists {table}")

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

    # Tabelas antigas mantidas por compatibilidade.
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

    # Tabelas novas: runs verdadeiramente separados por campo.
    for table, tokenizer in [
        ("docs_all", "unicode61 remove_diacritics 2"),
        ("docs_title", "unicode61 remove_diacritics 2"),
        ("docs_keywords", "unicode61 remove_diacritics 2"),
        ("docs_title_keywords", "unicode61 remove_diacritics 2"),
        ("docs_text", "unicode61 remove_diacritics 2"),
        ("docs_all_porter", "porter unicode61 remove_diacritics 2"),
        ("docs_title_porter", "porter unicode61 remove_diacritics 2"),
        ("docs_keywords_porter", "porter unicode61 remove_diacritics 2"),
        ("docs_title_keywords_porter", "porter unicode61 remove_diacritics 2"),
    ]:
        con.execute(
            f"""
            create virtual table {table} using fts5(
                entity_id unindexed,
                content,
                tokenize='{tokenizer}'
            )
            """
        )

    con.commit()


def insert_batch(con: sqlite3.Connection, batch: list[tuple[str, str, str, str, str, str]]) -> None:
    entity_rows = [(eid, title, keywords, text) for eid, title, keywords, text, _, _ in batch]
    con.executemany(
        "insert into entities(entity_id, title, keywords, text) values (?, ?, ?, ?)",
        entity_rows,
    )

    legacy_rows = [(eid, title, keywords, text) for eid, title, keywords, text, _, _ in batch]
    for table in ["docs", "docs_porter"]:
        con.executemany(
            f"insert into {table}(entity_id, title, keywords, text) values (?, ?, ?, ?)",
            legacy_rows,
        )

    all_rows = [(eid, all_content) for eid, _, _, _, _, all_content in batch]
    title_rows = [(eid, title) for eid, title, _, _, _, _ in batch]
    keywords_rows = [(eid, keywords) for eid, _, keywords, _, _, _ in batch]
    title_keywords_rows = [(eid, title_keywords) for eid, _, _, _, title_keywords, _ in batch]
    text_rows = [(eid, text) for eid, _, _, text, _, _ in batch]

    table_to_rows = {
        "docs_all": all_rows,
        "docs_title": title_rows,
        "docs_keywords": keywords_rows,
        "docs_title_keywords": title_keywords_rows,
        "docs_text": text_rows,
        "docs_all_porter": all_rows,
        "docs_title_porter": title_rows,
        "docs_keywords_porter": keywords_rows,
        "docs_title_keywords_porter": title_keywords_rows,
    }

    for table, rows in table_to_rows.items():
        con.executemany(
            f"insert into {table}(entity_id, content) values (?, ?)",
            rows,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Index only first N entities; useful for smoke tests.")
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
