from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from tqdm import tqdm


DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "entities.sqlite"
MODEL_PATH = ARTIFACTS_DIR / "ltr_model.pkl"
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
    operator: str = "OR"
    remove_stopwords: bool = False


VARIANTS = [
    Variant("and_balanced", "docs", (0.0, 4.0, 2.5, 0.8), operator="AND", remove_stopwords=True),
    Variant("and_title_keywords", "docs", (0.0, 6.0, 5.0, 0.4), operator="AND", remove_stopwords=True),
    Variant("porter_and_balanced", "docs_porter", (0.0, 4.0, 2.5, 0.8), operator="AND", remove_stopwords=True),
    Variant("or_title_heavy", "docs", (0.0, 9.0, 2.0, 0.4), remove_stopwords=True),
    Variant("or_title_keywords", "docs", (0.0, 5.0, 4.5, 0.5), remove_stopwords=True),
    Variant("or_keywords_heavy", "docs", (0.0, 2.0, 7.0, 0.5), remove_stopwords=True),
    Variant("porter_or_balanced", "docs_porter", (0.0, 4.0, 2.5, 0.8), remove_stopwords=True),
]


FEATURE_NAMES = [
    "rrf",
    "best_rank",
    "best_bm25",
    "variant_hits",
    "mean_rank",
    "title_cov",
    "keyword_cov",
    "text_cov",
    "title_keyword_cov",
    "title_hits",
    "keyword_hits",
    "text_hits",
    "title_bigram_cov",
    "keyword_bigram_cov",
    "phrase_in_title",
    "phrase_in_keywords",
    "phrase_in_text",
    "exact_title",
    "all_title",
    "all_title_keywords",
    "title_len",
    "keywords_len",
    "text_len",
    "short_title_good",
    "long_title_weak",
    "heuristic_bonus",
    *[f"{variant.name}_rank" for variant in VARIANTS],
    *[f"{variant.name}_bm25" for variant in VARIANTS],
]


def read_queries(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [(row["QueryId"], row["Query"]) for row in csv.DictReader(fh)]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            qrels[row["QueryId"]][row["EntityId"]] = int(row["Relevance"])
    return qrels


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    values = TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        filtered = [value for value in values if value not in STOPWORDS]
        return filtered or values
    return values


def fts_query(query: str, operator: str = "OR", remove_stopwords: bool = False) -> str:
    values = tokenize(query, remove_stopwords=remove_stopwords)
    if not values:
        return '""'
    joiner = f" {operator} "
    return joiner.join(f'"{value}"' for value in values)


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def bigrams(values: list[str]) -> set[tuple[str, str]]:
    return set(zip(values, values[1:]))


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(INDEX_PATH)
    con.execute("pragma temp_store=MEMORY")
    con.execute("pragma cache_size=-2000000")
    return con


def run_variant(con: sqlite3.Connection, query: str, variant: Variant, candidate_k: int) -> list[tuple[str, float, int]]:
    sql = f"""
        select entity_id, bm25({variant.table}, ?, ?, ?, ?) as score
        from {variant.table}
        where {variant.table} match ?
        order by score asc
        limit ?
    """
    expr = fts_query(query, operator=variant.operator, remove_stopwords=variant.remove_stopwords)
    try:
        rows = con.execute(sql, (*variant.weights, expr, candidate_k)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(entity_id, -float(score), rank) for rank, (entity_id, score) in enumerate(rows, start=1)]


def collect_candidates(
    con: sqlite3.Connection,
    query: str,
    candidate_k: int,
    rrf_k: int,
    adaptive_or_threshold: int,
) -> dict[str, dict[str, float]]:
    candidates: dict[str, dict[str, float]] = {}
    for variant_index, variant in enumerate(VARIANTS):
        if variant.operator == "OR" and len(candidates) >= adaptive_or_threshold:
            continue
        per_variant_k = candidate_k if variant.operator == "AND" else max(100, candidate_k // 2)
        for entity_id, bm25_score, rank in run_variant(con, query, variant, per_variant_k):
            item = candidates.setdefault(
                entity_id,
                {
                    "rrf": 0.0,
                    "best_bm25": 0.0,
                    "best_rank": float("inf"),
                    "variant_hits": 0.0,
                    "rank_sum": 0.0,
                    **{f"{v.name}_rank": 0.0 for v in VARIANTS},
                    **{f"{v.name}_bm25": 0.0 for v in VARIANTS},
                },
            )
            item["rrf"] += 1.0 / (rrf_k + rank)
            item["best_bm25"] = max(item["best_bm25"], bm25_score)
            item["best_rank"] = min(item["best_rank"], rank)
            item["variant_hits"] += 1.0
            item["rank_sum"] += rank
            item[f"{variant.name}_rank"] = float(rank)
            item[f"{variant.name}_bm25"] = bm25_score
    return candidates


def load_fields(con: sqlite3.Connection, entity_ids: list[str]) -> dict[str, tuple[str, str, str]]:
    fields: dict[str, tuple[str, str, str]] = {}
    for i in range(0, len(entity_ids), 500):
        chunk = entity_ids[i : i + 500]
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
    q_tokens = tokenize(query, remove_stopwords=True)
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
    title_bigram_cov = len(q_bigrams & bigrams(tokenize(title))) / bigram_denom
    keyword_bigram_cov = len(q_bigrams & bigrams(tokenize(keywords))) / bigram_denom
    title_len = len(title_terms)
    bonus = 0.34 * title_cov + 0.18 * keyword_cov + 0.04 * text_cov + 0.18 * title_keywords_cov
    bonus += 0.18 * title_bigram_cov + 0.08 * keyword_bigram_cov
    bonus += 0.35 if q and q in title.lower() else 0.0
    bonus += 0.28 if q_set <= title_terms else 0.0
    bonus += 0.18 if q_set <= title_keywords_terms else 0.0
    bonus += 0.12 if title_len <= 5 and title_cov >= 0.6 else 0.0
    bonus += 0.10 if title_len <= 8 and title_cov >= 0.8 else 0.0
    bonus -= 0.10 if title_len >= 18 and title_cov < 0.5 else 0.0
    bonus -= 0.08 if title_len >= 28 else 0.0
    return bonus


def make_features(query: str, stats: dict[str, float], fields: tuple[str, str, str]) -> list[float]:
    title, keywords, text = fields
    q = query.lower().strip()
    q_tokens = tokenize(query, remove_stopwords=True)
    q_set = set(q_tokens)
    denom = len(q_set) or 1
    title_terms = token_set(title)
    keyword_terms = token_set(keywords)
    text_terms = token_set(text)
    title_keywords_terms = title_terms | keyword_terms
    q_bigrams = bigrams(q_tokens)
    bigram_denom = len(q_bigrams) or 1
    title_hits = len(q_set & title_terms)
    keyword_hits = len(q_set & keyword_terms)
    text_hits = len(q_set & text_terms)
    title_cov = title_hits / denom
    keyword_cov = keyword_hits / denom
    text_cov = text_hits / denom
    title_keyword_cov = len(q_set & title_keywords_terms) / denom
    title_bigram_cov = len(q_bigrams & bigrams(tokenize(title))) / bigram_denom
    keyword_bigram_cov = len(q_bigrams & bigrams(tokenize(keywords))) / bigram_denom
    title_len = len(title_terms)
    keywords_len = len(keyword_terms)
    text_len = len(text_terms)
    values = [
        stats["rrf"],
        stats["best_rank"],
        stats["best_bm25"],
        stats["variant_hits"],
        stats["rank_sum"] / max(stats["variant_hits"], 1.0),
        title_cov,
        keyword_cov,
        text_cov,
        title_keyword_cov,
        float(title_hits),
        float(keyword_hits),
        float(text_hits),
        title_bigram_cov,
        keyword_bigram_cov,
        1.0 if q and q in title.lower() else 0.0,
        1.0 if q and q in keywords.lower() else 0.0,
        1.0 if q and q in text.lower() else 0.0,
        1.0 if q == title.lower().strip() else 0.0,
        1.0 if q_set and q_set <= title_terms else 0.0,
        1.0 if q_set and q_set <= title_keywords_terms else 0.0,
        float(title_len),
        float(keywords_len),
        float(text_len),
        1.0 if title_len <= 5 and title_cov >= 0.6 else 0.0,
        1.0 if title_len >= 18 and title_cov < 0.5 else 0.0,
        heuristic_bonus(query, fields),
    ]
    values.extend(stats[f"{variant.name}_rank"] for variant in VARIANTS)
    values.extend(stats[f"{variant.name}_bm25"] for variant in VARIANTS)
    return [float(value) for value in values]


def build_matrix(
    con: sqlite3.Connection,
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]] | None,
    candidate_k: int,
    rrf_k: int,
    adaptive_or_threshold: int,
    augment_qrels: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[tuple[str, str]], dict[str, list[str]]]:
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    keys: list[tuple[str, str]] = []
    pools: dict[str, list[str]] = {}
    for query_id, query in tqdm(queries, desc="features", unit="query"):
        candidates = collect_candidates(
            con,
            query,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            adaptive_or_threshold=adaptive_or_threshold,
        )
        if augment_qrels and qrels is not None:
            for entity_id in qrels.get(query_id, {}):
                candidates.setdefault(
                    entity_id,
                    {
                        "rrf": 0.0,
                        "best_bm25": 0.0,
                        "best_rank": float(candidate_k + 1),
                        "variant_hits": 0.0,
                        "rank_sum": 0.0,
                        **{f"{v.name}_rank": 0.0 for v in VARIANTS},
                        **{f"{v.name}_bm25": 0.0 for v in VARIANTS},
                    },
                )
        fields = load_fields(con, list(candidates))
        query_count = 0
        ranked_by_rrf = sorted(candidates, key=lambda eid: (-candidates[eid]["rrf"], candidates[eid]["best_rank"], eid))
        pools[query_id] = ranked_by_rrf
        for entity_id in ranked_by_rrf:
            if entity_id not in fields:
                continue
            X.append(make_features(query, candidates[entity_id], fields[entity_id]))
            if qrels is not None:
                y.append(qrels.get(query_id, {}).get(entity_id, 0))
            keys.append((query_id, entity_id))
            query_count += 1
        groups.append(query_count)
    y_array = np.asarray(y, dtype=np.int32) if qrels is not None else None
    return np.asarray(X, dtype=np.float32), y_array, groups, keys, pools


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


def run_from_scores(keys: list[tuple[str, str]], scores: np.ndarray, top_k: int) -> dict[str, list[str]]:
    by_query: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (query_id, entity_id), score in zip(keys, scores):
        by_query[query_id].append((entity_id, float(score)))
    return {
        query_id: [entity_id for entity_id, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]]
        for query_id, items in by_query.items()
    }


def rrf_run(pools: dict[str, list[str]], top_k: int) -> dict[str, list[str]]:
    return {query_id: entity_ids[:top_k] for query_id, entity_ids in pools.items()}


def fallback_ids(con: sqlite3.Connection, limit: int = 1000) -> list[str]:
    return [row[0] for row in con.execute("select entity_id from entities order by entity_id limit ?", (limit,))]


def oracle_recall(pools: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> tuple[float, float]:
    recalls = []
    weighted = []
    for query_id, rels in qrels.items():
        rel_ids = set(rels)
        top = set(pools.get(query_id, [])[:k])
        recalls.append(len(rel_ids & top) / len(rel_ids))
        total_gain = sum(2**rel - 1 for rel in rels.values())
        hit_gain = sum(2**rel - 1 for entity_id, rel in rels.items() if entity_id in top)
        weighted.append(hit_gain / total_gain if total_gain else 0.0)
    return sum(recalls) / len(recalls), sum(weighted) / len(weighted)


def split_queries(queries: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    train = [query for idx, query in enumerate(queries) if idx % 5 != 0]
    valid = [query for idx, query in enumerate(queries) if idx % 5 == 0]
    return train, valid


def train_model(X: np.ndarray, y: np.ndarray, groups: list[int]) -> lgb.Booster:
    dataset = lgb.Dataset(X, label=y, group=groups, feature_name=FEATURE_NAMES, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [100],
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "label_gain": [0, 1, 3],
        "seed": 13,
        "num_threads": -1,
        "verbosity": -1,
    }
    return lgb.train(params, dataset, num_boost_round=450)


def write_submission(
    path: Path,
    run: dict[str, list[str]],
    queries: list[tuple[str, str]],
    top_k: int,
    fillers: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["QueryId", "EntityId"])
        for query_id, _ in queries:
            entity_ids = list(run.get(query_id, [])[:top_k])
            seen = set(entity_ids)
            for entity_id in fillers or []:
                if len(entity_ids) >= top_k:
                    break
                if entity_id not in seen:
                    entity_ids.append(entity_id)
                    seen.add(entity_id)
            for entity_id in entity_ids[:top_k]:
                writer.writerow([query_id, entity_id])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "train", "submission"], default="eval")
    parser.add_argument("--candidate-k", type=int, default=800)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--adaptive-or-threshold", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "submission_ltr.csv")
    parser.add_argument("--augment-qrels", action="store_true")
    args = parser.parse_args()

    con = connect()
    qrels = read_qrels(DATA_DIR / "train_qrels.csv")
    train_queries = read_queries(DATA_DIR / "train_queries.csv")
    try:
        if args.mode == "eval":
            fit_queries, valid_queries = split_queries(train_queries)
            X_train, y_train, train_groups, _, _ = build_matrix(
                con,
                fit_queries,
                qrels,
                args.candidate_k,
                args.rrf_k,
                args.adaptive_or_threshold,
                augment_qrels=args.augment_qrels,
            )
            X_valid, _, _, valid_keys, valid_pools = build_matrix(
                con, valid_queries, qrels, args.candidate_k, args.rrf_k, args.adaptive_or_threshold
            )
            model = train_model(X_train, y_train, train_groups)
            valid_scores = model.predict(X_valid)
            valid_run = run_from_scores(valid_keys, valid_scores, args.top_k)
            base_run = rrf_run(valid_pools, args.top_k)
            valid_qrels = {query_id: qrels[query_id] for query_id, _ in valid_queries if query_id in qrels}
            recall100, gain100 = oracle_recall(valid_pools, valid_qrels, args.top_k)
            recall_pool, gain_pool = oracle_recall(valid_pools, valid_qrels, args.candidate_k)
            print(f"features: train={X_train.shape} valid={X_valid.shape}")
            print(f"rrf nDCG@{args.top_k}: {ndcg_at_k(base_run, valid_qrels, args.top_k):.6f}")
            print(f"ltr nDCG@{args.top_k}: {ndcg_at_k(valid_run, valid_qrels, args.top_k):.6f}")
            print(f"pool recall@{args.top_k}: {recall100:.6f} gain_recall@{args.top_k}: {gain100:.6f}")
            print(f"pool recall@{args.candidate_k}: {recall_pool:.6f} gain_recall@{args.candidate_k}: {gain_pool:.6f}")
        elif args.mode == "train":
            X, y, groups, _, _ = build_matrix(
                con,
                train_queries,
                qrels,
                args.candidate_k,
                args.rrf_k,
                args.adaptive_or_threshold,
                augment_qrels=args.augment_qrels,
            )
            model = train_model(X, y, groups)
            MODEL_PATH.parent.mkdir(exist_ok=True)
            with MODEL_PATH.open("wb") as fh:
                pickle.dump({"model": model, "features": FEATURE_NAMES, "candidate_k": args.candidate_k}, fh)
            print(f"trained {MODEL_PATH} with X={X.shape}")
        else:
            with MODEL_PATH.open("rb") as fh:
                payload = pickle.load(fh)
            test_queries = read_queries(DATA_DIR / "test_queries.csv")
            X, _, _, keys, _ = build_matrix(
                con, test_queries, None, args.candidate_k, args.rrf_k, args.adaptive_or_threshold
            )
            scores = payload["model"].predict(X)
            run = run_from_scores(keys, scores, args.top_k)
            write_submission(args.out, run, test_queries, args.top_k, fillers=fallback_ids(con))
            print(f"wrote {args.out}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
