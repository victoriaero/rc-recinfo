from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .index import connect
from .retrieve import RunRow, fts_query, read_run

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

FEATURE_NAMES = [
    "candidate_score",
    "bm25_title",
    "bm25_keywords",
    "bm25_text",
    "title_term_count",
    "keywords_term_count",
    "text_term_count",
    "title_coverage",
    "keywords_coverage",
    "text_coverage",
    "title_len",
    "keywords_len",
    "text_len",
    "title_exact",
    "title_partial",
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def generate_features(config: dict[str, Any], run_path: str | Path, output_path: str | Path, queries: dict[str, str]) -> Path:
    con = connect(config["paths"]["index_path"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_run(run_path)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["QueryId", "EntityId", "Rank", *FEATURE_NAMES])
        for row in tqdm(rows, desc="features", unit="pair"):
            query_text = queries[row.query_id]
            fields = con.execute(
                "select title, keywords, text from entities where entity_id = ?",
                (row.entity_id,),
            ).fetchone()
            if fields is None:
                continue
            writer.writerow([row.query_id, row.entity_id, row.rank, *feature_values(con, row, query_text, fields)])
    con.close()
    return output


def feature_values(con: sqlite3.Connection, row: RunRow, query_text: str, fields: tuple[str, str, str]) -> list[float]:
    title, keywords, text = fields
    terms = set(tokenize(query_text))
    denom = len(terms) or 1
    title_terms = set(tokenize(title))
    keyword_terms = set(tokenize(keywords))
    text_terms = set(tokenize(text))
    query_expr = fts_query(query_text)
    bm25_title = entity_bm25(con, row.entity_id, query_expr, (0, 1, 0, 0, 0))
    bm25_keywords = entity_bm25(con, row.entity_id, query_expr, (0, 0, 1, 0, 0))
    bm25_text = entity_bm25(con, row.entity_id, query_expr, (0, 0, 0, 1, 0))
    exact = 1.0 if query_text.lower().strip() == title.lower().strip() else 0.0
    partial = 1.0 if query_text.lower().strip() in title.lower() else 0.0
    return [
        row.score,
        bm25_title,
        bm25_keywords,
        bm25_text,
        float(len(terms & title_terms)),
        float(len(terms & keyword_terms)),
        float(len(terms & text_terms)),
        float(len(terms & title_terms) / denom),
        float(len(terms & keyword_terms) / denom),
        float(len(terms & text_terms) / denom),
        float(len(title_terms)),
        float(len(keyword_terms)),
        float(len(text_terms)),
        exact,
        partial,
    ]


def entity_bm25(
    con: sqlite3.Connection,
    entity_id: str,
    query_expr: str,
    weights: tuple[float, float, float, float, float],
) -> float:
    row = con.execute(
        """
        select bm25(docs, ?, ?, ?, ?, ?) as score
        from docs
        where docs match ? and entity_id = ?
        limit 1
        """,
        (*weights, query_expr, entity_id),
    ).fetchone()
    return float(row[0]) if row is not None else 0.0
