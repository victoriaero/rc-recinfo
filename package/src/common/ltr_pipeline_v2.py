from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

# Supervised lexical Learning-to-Rank pipeline built on BM25/RRF candidates.
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")
INDEX_PATH = ARTIFACTS_DIR / "entities.sqlite"
MODEL_PATH = ARTIFACTS_DIR / "ltr_model_v2.pkl"

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


@dataclass(frozen=True)
class Variant:
    name: str
    table: str
    operator: str = "OR"
    weight: float = 1.0
    per_variant_factor: float = 1.0


# Stopwords are intentionally kept; diversity comes from field, operator, and stemming.
VARIANTS = [
    Variant("title_and", "docs_title", operator="AND", weight=2.7, per_variant_factor=1.0),
    Variant("title_keywords_and", "docs_title_keywords", operator="AND", weight=2.4, per_variant_factor=1.0),
    Variant("title_or", "docs_title", operator="OR", weight=2.1, per_variant_factor=0.75),
    Variant("title_keywords_or", "docs_title_keywords", operator="OR", weight=1.9, per_variant_factor=0.75),
    Variant("keywords_and", "docs_keywords", operator="AND", weight=1.6, per_variant_factor=1.0),
    Variant("keywords_or", "docs_keywords", operator="OR", weight=1.3, per_variant_factor=0.7),
    Variant("all_and", "docs_all", operator="AND", weight=1.2, per_variant_factor=1.0),
    Variant("all_or", "docs_all", operator="OR", weight=0.9, per_variant_factor=0.6),
    Variant("porter_title_keywords_or", "docs_title_keywords_porter", operator="OR", weight=0.9, per_variant_factor=0.65),
    Variant("porter_all_or", "docs_all_porter", operator="OR", weight=0.7, per_variant_factor=0.5),
    Variant("text_or", "docs_text", operator="OR", weight=0.25, per_variant_factor=0.35),
]

FEATURE_NAMES = [
    "rrf",
    "best_rank",
    "inv_best_rank",
    "mean_rank",
    "inv_mean_rank",
    "best_bm25",
    "log_best_bm25",
    "variant_hits",
    "log_variant_hits",
    "title_cov",
    "keyword_cov",
    "text_cov",
    "title_keyword_cov",
    "title_jaccard",
    "keyword_jaccard",
    "title_keyword_jaccard",
    "title_hits",
    "keyword_hits",
    "text_hits",
    "title_bigram_cov",
    "keyword_bigram_cov",
    "title_keyword_bigram_cov",
    "any_bigram_title",
    "any_bigram_keywords",
    "phrase_in_title",
    "phrase_in_keywords",
    "phrase_in_text",
    "exact_title",
    "all_title",
    "all_keywords",
    "all_title_keywords",
    "first_term_in_title",
    "last_term_in_title",
    "ordered_terms_in_title",
    "title_len",
    "keywords_len",
    "text_len",
    "title_len_chars",
    "short_title_good",
    "very_short_exactish",
    "long_title_weak",
    "parentheses_in_title",
    "heuristic_bonus",
    "max_title_family_bm25",
    "max_keywords_family_bm25",
    "max_title_keywords_family_bm25",
    "min_title_family_rank",
    "min_title_keywords_family_rank",
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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    # Keep stopwords so phrase-like entity queries preserve their original signal.
    return TOKEN_RE.findall(normalize_text(text))


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def bigrams(values: list[str]) -> set[tuple[str, str]]:
    return set(zip(values, values[1:]))


def ordered_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    pos = 0
    for token in haystack:
        if token == needle[pos]:
            pos += 1
            if pos == len(needle):
                return True
    return False


def fts_query(query: str, operator: str = "OR") -> str:
    values = tokenize(query)
    if not values:
        return '""'
    joiner = f" {operator} "
    return joiner.join(f'"{value}"' for value in values)


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(INDEX_PATH)
    con.execute("pragma temp_store=MEMORY")
    con.execute("pragma cache_size=-2000000")
    return con


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select name from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()
    return row is not None


def validate_index(con: sqlite3.Connection) -> None:
    missing = [variant.table for variant in VARIANTS if not table_exists(con, variant.table)]
    missing = sorted(set(missing))
    if missing:
        raise RuntimeError(
            "Missing FTS tables in artifacts/entities.sqlite: "
            + ", ".join(missing)
            + ". Rebuild the index with build_index_v2.py first."
        )


def run_variant(con: sqlite3.Connection, query: str, variant: Variant, candidate_k: int) -> list[tuple[str, float, int]]:
    per_variant_k = max(50, int(candidate_k * variant.per_variant_factor))
    expr = fts_query(query, operator=variant.operator)

    sql = f"""
        select entity_id, bm25({variant.table}) as score
        from {variant.table}
        where {variant.table} match ?
        order by score asc
        limit ?
    """

    try:
        rows = con.execute(sql, (expr, per_variant_k)).fetchall()
    except sqlite3.OperationalError:
        return []

    return [(entity_id, -float(score), rank) for rank, (entity_id, score) in enumerate(rows, start=1)]


def empty_stats(candidate_k: int) -> dict[str, float]:
    return {
        "rrf": 0.0,
        "best_bm25": 0.0,
        "best_rank": float(candidate_k + 1),
        "variant_hits": 0.0,
        "rank_sum": 0.0,
        **{f"{v.name}_rank": 0.0 for v in VARIANTS},
        **{f"{v.name}_bm25": 0.0 for v in VARIANTS},
    }


def collect_candidates(
    con: sqlite3.Connection,
    query: str,
    candidate_k: int,
    rrf_k: int,
) -> dict[str, dict[str, float]]:
    candidates: dict[str, dict[str, float]] = {}

    for variant in VARIANTS:
        for entity_id, bm25_score, rank in run_variant(con, query, variant, candidate_k):
            item = candidates.setdefault(entity_id, empty_stats(candidate_k))
            # Variant weights tune the contribution of each field/operator view.
            item["rrf"] += variant.weight / (rrf_k + rank)
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
        if not chunk:
            continue
        marks = ",".join("?" for _ in chunk)
        for entity_id, title, keywords, text in con.execute(
            f"select entity_id, title, keywords, text from entities where entity_id in ({marks})",
            chunk,
        ):
            fields[entity_id] = (title or "", keywords or "", text or "")
    return fields


def safe_min_positive(values: list[float], default: float) -> float:
    positives = [value for value in values if value > 0]
    return min(positives) if positives else default


def heuristic_bonus(query: str, fields: tuple[str, str, str]) -> float:
    title, keywords, text = fields
    q_norm = normalize_text(query)
    q_tokens = tokenize(query)
    q_set = set(q_tokens)

    if not q_set:
        return 0.0

    title_tokens = tokenize(title)
    keyword_tokens = tokenize(keywords)
    text_tokens = tokenize(text)

    title_terms = set(title_tokens)
    keyword_terms = set(keyword_tokens)
    text_terms = set(text_tokens)
    title_keywords_terms = title_terms | keyword_terms

    denom = len(q_set)
    title_cov = len(q_set & title_terms) / denom
    keyword_cov = len(q_set & keyword_terms) / denom
    text_cov = len(q_set & text_terms) / denom
    title_keyword_cov = len(q_set & title_keywords_terms) / denom

    q_bigrams = bigrams(q_tokens)
    bigram_denom = len(q_bigrams) or 1
    title_bigram_cov = len(q_bigrams & bigrams(title_tokens)) / bigram_denom
    keyword_bigram_cov = len(q_bigrams & bigrams(keyword_tokens)) / bigram_denom

    title_norm = normalize_text(title)
    keywords_norm = normalize_text(keywords)
    title_len = len(title_tokens)

    bonus = 0.42 * title_cov
    bonus += 0.22 * keyword_cov
    bonus += 0.04 * text_cov
    bonus += 0.24 * title_keyword_cov
    bonus += 0.28 * title_bigram_cov
    bonus += 0.12 * keyword_bigram_cov

    bonus += 0.55 if q_norm and q_norm in title_norm else 0.0
    bonus += 0.25 if q_norm and q_norm in keywords_norm else 0.0
    bonus += 0.35 if q_set <= title_terms else 0.0
    bonus += 0.24 if q_set <= title_keywords_terms else 0.0
    bonus += 0.16 if title_len <= 5 and title_cov >= 0.6 else 0.0
    bonus += 0.14 if title_len <= 8 and title_cov >= 0.8 else 0.0

    bonus -= 0.14 if title_len >= 18 and title_cov < 0.5 else 0.0
    bonus -= 0.10 if title_len >= 28 else 0.0

    return bonus


def make_features(query: str, stats: dict[str, float], fields: tuple[str, str, str]) -> list[float]:
    title, keywords, text = fields

    # Feature groups mix retrieval evidence, lexical overlap, and entity shape.
    q_norm = normalize_text(query)
    q_tokens = tokenize(query)
    q_set = set(q_tokens)
    denom = len(q_set) or 1

    title_tokens = tokenize(title)
    keyword_tokens = tokenize(keywords)
    text_tokens = tokenize(text)

    title_terms = set(title_tokens)
    keyword_terms = set(keyword_tokens)
    text_terms = set(text_tokens)
    title_keywords_terms = title_terms | keyword_terms

    q_bigrams = bigrams(q_tokens)
    title_bigrams = bigrams(title_tokens)
    keyword_bigrams = bigrams(keyword_tokens)
    title_keyword_bigrams = title_bigrams | keyword_bigrams
    bigram_denom = len(q_bigrams) or 1

    title_hits = len(q_set & title_terms)
    keyword_hits = len(q_set & keyword_terms)
    text_hits = len(q_set & text_terms)

    title_cov = title_hits / denom
    keyword_cov = keyword_hits / denom
    text_cov = text_hits / denom
    title_keyword_cov = len(q_set & title_keywords_terms) / denom

    title_jaccard = len(q_set & title_terms) / (len(q_set | title_terms) or 1)
    keyword_jaccard = len(q_set & keyword_terms) / (len(q_set | keyword_terms) or 1)
    title_keyword_jaccard = len(q_set & title_keywords_terms) / (len(q_set | title_keywords_terms) or 1)

    title_bigram_cov = len(q_bigrams & title_bigrams) / bigram_denom
    keyword_bigram_cov = len(q_bigrams & keyword_bigrams) / bigram_denom
    title_keyword_bigram_cov = len(q_bigrams & title_keyword_bigrams) / bigram_denom

    title_norm = normalize_text(title)
    keywords_norm = normalize_text(keywords)
    text_norm = normalize_text(text)

    title_len = len(title_tokens)
    keywords_len = len(keyword_tokens)
    text_len = len(text_tokens)
    best_rank = stats["best_rank"]
    mean_rank = stats["rank_sum"] / max(stats["variant_hits"], 1.0)

    title_family_bm25 = [
        stats.get("title_and_bm25", 0.0),
        stats.get("title_or_bm25", 0.0),
    ]
    keywords_family_bm25 = [
        stats.get("keywords_and_bm25", 0.0),
        stats.get("keywords_or_bm25", 0.0),
    ]
    title_keywords_family_bm25 = [
        stats.get("title_keywords_and_bm25", 0.0),
        stats.get("title_keywords_or_bm25", 0.0),
        stats.get("porter_title_keywords_or_bm25", 0.0),
    ]

    title_family_ranks = [
        stats.get("title_and_rank", 0.0),
        stats.get("title_or_rank", 0.0),
    ]
    title_keywords_family_ranks = [
        stats.get("title_keywords_and_rank", 0.0),
        stats.get("title_keywords_or_rank", 0.0),
        stats.get("porter_title_keywords_or_rank", 0.0),
    ]

    values = [
        stats["rrf"],
        best_rank,
        1.0 / max(best_rank, 1.0),
        mean_rank,
        1.0 / max(mean_rank, 1.0),
        stats["best_bm25"],
        math.log1p(max(stats["best_bm25"], 0.0)),
        stats["variant_hits"],
        math.log1p(stats["variant_hits"]),
        title_cov,
        keyword_cov,
        text_cov,
        title_keyword_cov,
        title_jaccard,
        keyword_jaccard,
        title_keyword_jaccard,
        float(title_hits),
        float(keyword_hits),
        float(text_hits),
        title_bigram_cov,
        keyword_bigram_cov,
        title_keyword_bigram_cov,
        1.0 if q_bigrams and bool(q_bigrams & title_bigrams) else 0.0,
        1.0 if q_bigrams and bool(q_bigrams & keyword_bigrams) else 0.0,
        1.0 if q_norm and q_norm in title_norm else 0.0,
        1.0 if q_norm and q_norm in keywords_norm else 0.0,
        1.0 if q_norm and q_norm in text_norm else 0.0,
        1.0 if q_norm == title_norm else 0.0,
        1.0 if q_set and q_set <= title_terms else 0.0,
        1.0 if q_set and q_set <= keyword_terms else 0.0,
        1.0 if q_set and q_set <= title_keywords_terms else 0.0,
        1.0 if q_tokens and q_tokens[0] in title_terms else 0.0,
        1.0 if q_tokens and q_tokens[-1] in title_terms else 0.0,
        1.0 if ordered_subsequence(q_tokens, title_tokens) else 0.0,
        float(title_len),
        float(keywords_len),
        float(text_len),
        float(len(title)),
        1.0 if title_len <= 5 and title_cov >= 0.6 else 0.0,
        1.0 if title_len <= 4 and title_cov >= 0.8 else 0.0,
        1.0 if title_len >= 18 and title_cov < 0.5 else 0.0,
        1.0 if "(" in title and ")" in title else 0.0,
        heuristic_bonus(query, fields),
        max(title_family_bm25),
        max(keywords_family_bm25),
        max(title_keywords_family_bm25),
        safe_min_positive(title_family_ranks, default=0.0),
        safe_min_positive(title_keywords_family_ranks, default=0.0),
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
    augment_qrels: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[tuple[str, str]], dict[str, list[str]]]:
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    keys: list[tuple[str, str]] = []
    pools: dict[str, list[str]] = {}

    for query_id, query in tqdm(queries, desc="features", unit="query"):
        candidates = collect_candidates(con, query, candidate_k=candidate_k, rrf_k=rrf_k)

        if augment_qrels and qrels is not None:
            for entity_id in qrels.get(query_id, {}):
                candidates.setdefault(entity_id, empty_stats(candidate_k))

        # Keep the pool bounded; very large pools add mostly weak negatives.
        ranked_by_rrf = sorted(
            candidates,
            key=lambda eid: (-candidates[eid]["rrf"], candidates[eid]["best_rank"], eid),
        )[:candidate_k]

        fields = load_fields(con, ranked_by_rrf)
        pools[query_id] = ranked_by_rrf

        query_count = 0
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
    return sum(values) / len(values) if values else 0.0


def run_from_scores(keys: list[tuple[str, str]], scores: np.ndarray, top_k: int) -> dict[str, list[str]]:
    by_query: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (query_id, entity_id), score in zip(keys, scores):
        by_query[query_id].append((entity_id, float(score)))

    return {
        query_id: [
            entity_id
            for entity_id, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:top_k]
        ]
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


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: list[int],
    X_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    valid_groups: list[int] | None = None,
    num_boost_round: int = 1200,
    early_stopping_rounds: int | None = 100,
) -> lgb.Booster:
    # LambdaMART optimizes the top-100 ranking directly with graded relevance.
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_groups,
        feature_name=FEATURE_NAMES,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [100],
        "learning_rate": 0.03,
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

    callbacks = [lgb.log_evaluation(50)]

    valid_sets = None
    valid_names = None
    if X_valid is not None and y_valid is not None and valid_groups is not None:
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            group=valid_groups,
            feature_name=FEATURE_NAMES,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        valid_names = ["valid"]
        if early_stopping_rounds is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

    return lgb.train(
        params,
        train_set,
        valid_sets=valid_sets,
        valid_names=valid_names,
        num_boost_round=num_boost_round,
        callbacks=callbacks,
    )


def split_queries(queries: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    train = [query for idx, query in enumerate(queries) if idx % 5 != 0]
    valid = [query for idx, query in enumerate(queries) if idx % 5 == 0]
    return train, valid


def cv_eval(
    con: sqlite3.Connection,
    queries: list[tuple[str, str]],
    qrels: dict[str, dict[str, int]],
    candidate_k: int,
    rrf_k: int,
    top_k: int,
    n_splits: int,
    augment_qrels: bool,
) -> None:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=13)
    query_array = np.asarray(queries, dtype=object)

    rrf_scores = []
    ltr_scores = []
    recall_scores = []
    gain_scores = []
    best_iterations = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(query_array), start=1):
        fit_queries = [tuple(item) for item in query_array[train_idx]]
        valid_queries = [tuple(item) for item in query_array[valid_idx]]

        X_train, y_train, train_groups, _, _ = build_matrix(
            con, fit_queries, qrels, candidate_k, rrf_k, augment_qrels=augment_qrels
        )
        X_valid, y_valid, valid_groups, valid_keys, valid_pools = build_matrix(
            con, valid_queries, qrels, candidate_k, rrf_k, augment_qrels=False
        )

        model = train_model(
            X_train,
            y_train,
            train_groups,
            X_valid,
            y_valid,
            valid_groups,
            num_boost_round=1500,
            early_stopping_rounds=100,
        )

        scores = model.predict(X_valid, num_iteration=model.best_iteration)
        ltr_run = run_from_scores(valid_keys, scores, top_k)
        base_run = rrf_run(valid_pools, top_k)

        valid_qrels = {query_id: qrels[query_id] for query_id, _ in valid_queries if query_id in qrels}
        rrf_ndcg = ndcg_at_k(base_run, valid_qrels, top_k)
        ltr_ndcg = ndcg_at_k(ltr_run, valid_qrels, top_k)
        recall_pool, gain_pool = oracle_recall(valid_pools, valid_qrels, candidate_k)

        rrf_scores.append(rrf_ndcg)
        ltr_scores.append(ltr_ndcg)
        recall_scores.append(recall_pool)
        gain_scores.append(gain_pool)
        best_iterations.append(model.best_iteration or 1500)

        print(
            f"fold={fold} "
            f"rrf_nDCG@{top_k}={rrf_ndcg:.6f} "
            f"ltr_nDCG@{top_k}={ltr_ndcg:.6f} "
            f"pool_recall@{candidate_k}={recall_pool:.6f} "
            f"gain_recall@{candidate_k}={gain_pool:.6f} "
            f"best_iter={model.best_iteration}"
        )

    print("==== CV summary ====")
    print(f"rrf mean nDCG@{top_k}: {np.mean(rrf_scores):.6f} ± {np.std(rrf_scores):.6f}")
    print(f"ltr mean nDCG@{top_k}: {np.mean(ltr_scores):.6f} ± {np.std(ltr_scores):.6f}")
    print(f"pool recall@{candidate_k}: {np.mean(recall_scores):.6f}")
    print(f"gain recall@{candidate_k}: {np.mean(gain_scores):.6f}")
    print(f"recommended num_boost_round: {int(np.mean(best_iterations))}")


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
    parser.add_argument("--mode", choices=["eval", "cv", "train", "submission"], default="eval")
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ARTIFACTS_DIR / "submission_ltr_v2.csv")
    parser.add_argument("--augment-qrels", action="store_true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=600)
    args = parser.parse_args()

    con = connect()
    validate_index(con)

    qrels = read_qrels(DATA_DIR / "train_qrels.csv")
    train_queries = read_queries(DATA_DIR / "train_queries.csv")

    try:
        if args.mode == "cv":
            cv_eval(
                con,
                train_queries,
                qrels,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                top_k=args.top_k,
                n_splits=args.n_splits,
                augment_qrels=args.augment_qrels,
            )

        elif args.mode == "eval":
            fit_queries, valid_queries = split_queries(train_queries)

            X_train, y_train, train_groups, _, _ = build_matrix(
                con,
                fit_queries,
                qrels,
                args.candidate_k,
                args.rrf_k,
                augment_qrels=args.augment_qrels,
            )
            X_valid, y_valid, valid_groups, valid_keys, valid_pools = build_matrix(
                con,
                valid_queries,
                qrels,
                args.candidate_k,
                args.rrf_k,
                augment_qrels=False,
            )

            model = train_model(
                X_train,
                y_train,
                train_groups,
                X_valid,
                y_valid,
                valid_groups,
                num_boost_round=1500,
                early_stopping_rounds=100,
            )

            valid_scores = model.predict(X_valid, num_iteration=model.best_iteration)
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
            print(f"best_iteration: {model.best_iteration}")

        elif args.mode == "train":
            X, y, groups, _, _ = build_matrix(
                con,
                train_queries,
                qrels,
                args.candidate_k,
                args.rrf_k,
                augment_qrels=args.augment_qrels,
            )

            model = train_model(
                X,
                y,
                groups,
                num_boost_round=args.num_boost_round,
                early_stopping_rounds=None,
            )

            MODEL_PATH.parent.mkdir(exist_ok=True)
            payload = {
                "model": model,
                "features": FEATURE_NAMES,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "top_k": args.top_k,
                "variants": [variant.__dict__ for variant in VARIANTS],
                "stopwords": "none",
                "num_boost_round": args.num_boost_round,
            }

            with MODEL_PATH.open("wb") as fh:
                pickle.dump(payload, fh)

            print(f"trained {MODEL_PATH} with X={X.shape}, candidate_k={args.candidate_k}")

        else:
            with MODEL_PATH.open("rb") as fh:
                payload = pickle.load(fh)

            model_candidate_k = int(payload.get("candidate_k", args.candidate_k))
            model_rrf_k = int(payload.get("rrf_k", args.rrf_k))
            model_top_k = int(payload.get("top_k", args.top_k))

            if args.candidate_k != model_candidate_k:
                print(
                    f"warning: ignoring --candidate-k={args.candidate_k}; "
                    f"using model candidate_k={model_candidate_k}"
                )

            test_queries = read_queries(DATA_DIR / "test_queries.csv")
            X, _, _, keys, _ = build_matrix(
                con,
                test_queries,
                None,
                model_candidate_k,
                model_rrf_k,
                augment_qrels=False,
            )

            scores = payload["model"].predict(X)
            run = run_from_scores(keys, scores, model_top_k)
            write_submission(args.out, run, test_queries, model_top_k, fillers=fallback_ids(con))
            print(f"wrote {args.out}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
