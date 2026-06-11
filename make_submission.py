from __future__ import annotations

import argparse
import csv
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "entities.sqlite"
SUBMISSION_PATH = ARTIFACTS_DIR / "submission.csv"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "it", "of", "on", "or", "that", "the", "to", "with", "about", "after", "before", "during",
}


@dataclass(frozen=True)
class Variant:
    name: str
    table: str
    weights: tuple[float, float, float, float]
    remove_stopwords: bool = False


VARIANTS = [
    Variant("balanced", "docs", (0.0, 4.0, 2.5, 0.8)),
    Variant("title_heavy", "docs", (0.0, 8.0, 2.0, 0.5)),
    Variant("title_keywords", "docs", (0.0, 5.0, 4.0, 0.6)),
    Variant("keywords_heavy", "docs", (0.0, 2.5, 6.0, 0.6)),
    Variant("porter_balanced", "docs_porter", (0.0, 4.0, 2.5, 0.8)),
    Variant("porter_title_keywords", "docs_porter", (0.0, 5.0, 4.0, 0.6), remove_stopwords=True),
]


def read_queries(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [(row["QueryId"], row["Query"]) for row in csv.DictReader(fh)]


def tokens(query: str, remove_stopwords: bool = False) -> list[str]:
    values = TOKEN_RE.findall(query.lower())
    if remove_stopwords:
        filtered = [token for token in values if token not in STOPWORDS]
        return filtered or values
    return values


def fts_query(query: str, remove_stopwords: bool = False) -> str:
    values = tokens(query, remove_stopwords)
    return " OR ".join(f'"{token}"' for token in values) if values else '""'


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def bigrams(values: list[str]) -> set[tuple[str, str]]:
    return set(zip(values, values[1:]))


def run_variant(con: sqlite3.Connection, query: str, variant: Variant, candidate_k: int) -> list[tuple[str, float, int]]:
    table = variant.table
    sql = f"""
        select entity_id, bm25({table}, ?, ?, ?, ?) as score
        from {table}
        where {table} match ?
        order by score asc
        limit ?
    """
    rows = con.execute(sql, (*variant.weights, fts_query(query, variant.remove_stopwords), candidate_k)).fetchall()
    return [(entity_id, float(score), rank) for rank, (entity_id, score) in enumerate(rows, start=1)]


def collect_candidates(con: sqlite3.Connection, query: str, candidate_k: int, rrf_k: int) -> dict[str, dict[str, float]]:
    candidates: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        try:
            rows = run_variant(con, query, variant, candidate_k)
        except sqlite3.OperationalError:
            rows = []
        for entity_id, bm25_score, rank in rows:
            item = candidates.setdefault(
                entity_id,
                {
                    "rrf": 0.0,
                    "best_bm25": 0.0,
                    "best_rank": float("inf"),
                    "variant_hits": 0.0,
                },
            )
            item["rrf"] += 1.0 / (rrf_k + rank)
            item["best_bm25"] = max(item["best_bm25"], -bm25_score)
            item["best_rank"] = min(item["best_rank"], rank)
            item["variant_hits"] += 1.0
    return candidates


def load_fields(con: sqlite3.Connection, entity_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    if not entity_ids:
        return {}
    fields = {}
    chunk_size = 500
    for i in range(0, len(entity_ids), chunk_size):
        chunk = entity_ids[i : i + chunk_size]
        marks = ",".join("?" for _ in chunk)
        for entity_id, title, keywords, text in con.execute(
            f"select entity_id, title, keywords, text from entities where entity_id in ({marks})",
            chunk,
        ):
            fields[entity_id] = (title, keywords, text)
    return fields


def heuristic_bonus(query: str, fields: tuple[str, str, str]) -> float:
    title, keywords, text = fields
    q = query.lower().strip()
    q_tokens = tokens(query, remove_stopwords=True)
    q_set = set(q_tokens)
    if not q_set:
        return 0.0

    title_terms = token_set(title)
    keyword_terms = token_set(keywords)
    text_terms = token_set(text)
    title_keywords_terms = title_terms | keyword_terms
    denom = len(q_set)

    title_cov = len(q_set & title_terms) / denom
    keyword_cov = len(q_set & keyword_terms) / denom
    text_cov = len(q_set & text_terms) / denom
    title_keywords_cov = len(q_set & title_keywords_terms) / denom

    q_bigrams = bigrams(q_tokens)
    bigram_denom = len(q_bigrams) or 1
    title_bigram_cov = len(q_bigrams & bigrams(tokens(title))) / bigram_denom
    keyword_bigram_cov = len(q_bigrams & bigrams(tokens(keywords))) / bigram_denom
    title_len = len(title_terms)

    bonus = 0.0
    bonus += 0.34 * title_cov
    bonus += 0.18 * keyword_cov
    bonus += 0.04 * text_cov
    bonus += 0.18 * title_keywords_cov
    bonus += 0.18 * title_bigram_cov
    bonus += 0.08 * keyword_bigram_cov
    if q and q in title.lower():
        bonus += 0.35
    if q_set <= title_terms:
        bonus += 0.28
    if q_set <= title_keywords_terms:
        bonus += 0.18
    if title_len <= 5 and title_cov >= 0.6:
        bonus += 0.12
    if title_len <= 8 and title_cov >= 0.8:
        bonus += 0.10
    if title_len >= 18 and title_cov < 0.5:
        bonus -= 0.10
    if title_len >= 28:
        bonus -= 0.08
    return bonus


def final_score(query: str, stats: dict[str, float], fields: tuple[str, str, str]) -> float:
    rank_signal = 1.0 / math.log2(stats["best_rank"] + 2.0)
    return (
        10.0 * stats["rrf"]
        + 0.25 * stats["best_bm25"]
        + 0.08 * stats["variant_hits"]
        + 0.20 * rank_signal
        + heuristic_bonus(query, fields)
    )


def retrieve(con: sqlite3.Connection, query: str, candidate_k: int, top_k: int, rrf_k: int) -> list[str]:
    candidates = collect_candidates(con, query, candidate_k, rrf_k)
    fields = load_fields(con, list(candidates))
    scored = []
    for entity_id, stats in candidates.items():
        if entity_id not in fields:
            continue
        scored.append((entity_id, final_score(query, stats, fields[entity_id])))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [entity_id for entity_id, _ in scored[:top_k]]


def fallback_ids(con: sqlite3.Connection, limit: int = 1000) -> list[str]:
    return [row[0] for row in con.execute("select entity_id from entities order by entity_id limit ?", (limit,))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "test_queries.csv")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--out", type=Path, default=SUBMISSION_PATH)
    args = parser.parse_args()

    queries = read_queries(args.queries)
    con = sqlite3.connect(INDEX_PATH)
    con.execute("pragma temp_store=MEMORY")
    con.execute("pragma cache_size=-2000000")
    fillers = fallback_ids(con)

    args.out.parent.mkdir(exist_ok=True)
    tmp_out = args.out.with_suffix(args.out.suffix + ".tmp")
    total_predictions = 0
    completed_queries = 0
    try:
        with tmp_out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["QueryId", "EntityId"])
            for query_id, query in tqdm(queries, desc="retrieving", unit="query"):
                entity_ids = retrieve(con, query, args.candidate_k, args.top_k, args.rrf_k)
                seen = set(entity_ids)
                for entity_id in fillers:
                    if len(entity_ids) >= args.top_k:
                        break
                    if entity_id not in seen:
                        entity_ids.append(entity_id)
                        seen.add(entity_id)
                for entity_id in entity_ids[: args.top_k]:
                    writer.writerow([query_id, entity_id])
                    total_predictions += 1
                completed_queries += 1
        tmp_out.replace(args.out)
    finally:
        con.close()

    print(f"Completed {completed_queries}/{len(queries)} queries")
    print(f"Wrote {total_predictions + 1} lines including header to {args.out}")


if __name__ == "__main__":
    main()
