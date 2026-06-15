from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

DATA_DIR = Path('data')
ARTIFACTS_DIR = Path('artifacts')
CACHE_DIR = ARTIFACTS_DIR / 'cache_v3'
INDEX_PATH = ARTIFACTS_DIR / 'entities.sqlite'
MODEL_PATH = ARTIFACTS_DIR / 'ltr_model_v3.pkl'
TOKEN_RE = re.compile(r'[\w]+', re.UNICODE)


@dataclass(frozen=True)
class Variant:
    name: str
    table: str
    operator: str = 'OR'
    weight: float = 1.0
    per_variant_factor: float = 1.0
    query_mode: str = 'original'  # original | expanded
    weights: tuple[float, float, float] | None = None  # legacy docs/docs_porter title/keywords/text


BASE_VARIANTS = [
    Variant('title_and', 'docs_title', 'AND', 2.7, 1.0),
    Variant('title_keywords_and', 'docs_title_keywords', 'AND', 2.4, 1.0),
    Variant('title_or', 'docs_title', 'OR', 2.1, 0.75),
    Variant('title_keywords_or', 'docs_title_keywords', 'OR', 1.9, 0.75),
    Variant('keywords_and', 'docs_keywords', 'AND', 1.6, 1.0),
    Variant('keywords_or', 'docs_keywords', 'OR', 1.3, 0.7),
    Variant('all_and', 'docs_all', 'AND', 1.2, 1.0),
    Variant('all_or', 'docs_all', 'OR', 0.9, 0.6),
    Variant('porter_title_keywords_or', 'docs_title_keywords_porter', 'OR', 0.9, 0.65),
    Variant('porter_all_or', 'docs_all_porter', 'OR', 0.7, 0.5),
    Variant('text_or', 'docs_text', 'OR', 0.25, 0.35),
    Variant('legacy_balanced', 'docs', 'OR', 1.0, 0.75, weights=(1.0, 1.0, 1.0)),
    Variant('legacy_title_heavy', 'docs', 'OR', 1.45, 0.85, weights=(4.0, 1.5, 0.7)),
    Variant('legacy_title_keywords', 'docs', 'OR', 1.55, 0.9, weights=(3.0, 2.5, 0.6)),
    Variant('legacy_keywords_heavy', 'docs', 'OR', 1.05, 0.75, weights=(1.5, 4.0, 0.5)),
    Variant('legacy_porter_title_keywords', 'docs_porter', 'OR', 0.95, 0.75, weights=(3.0, 2.5, 0.6)),
]
EXPANDED_VARIANTS = [
    Variant('expanded_title_keywords_or', 'docs_title_keywords', 'OR', 0.8, 0.6, 'expanded'),
    Variant('expanded_all_or', 'docs_all', 'OR', 0.55, 0.45, 'expanded'),
    Variant('expanded_porter_title_keywords', 'docs_title_keywords_porter', 'OR', 0.55, 0.45, 'expanded'),
    Variant('expanded_legacy_title_keywords', 'docs', 'OR', 0.7, 0.5, 'expanded', weights=(3.0, 2.5, 0.5)),
]
VARIANTS = BASE_VARIANTS + EXPANDED_VARIANTS

LEXICAL_FEATURES = [
    'rrf','best_rank','inv_best_rank','mean_rank','inv_mean_rank','best_bm25','log_best_bm25',
    'variant_hits','log_variant_hits','title_cov','keyword_cov','text_cov','title_keyword_cov',
    'title_jaccard','keyword_jaccard','title_keyword_jaccard','title_hits','keyword_hits','text_hits',
    'title_bigram_cov','keyword_bigram_cov','title_keyword_bigram_cov','any_bigram_title','any_bigram_keywords',
    'phrase_in_title','phrase_in_keywords','phrase_in_text','exact_title','all_title','all_keywords','all_title_keywords',
    'first_term_in_title','last_term_in_title','ordered_terms_in_title','title_len','keywords_len','text_len',
    'title_len_chars','short_title_good','very_short_exactish','long_title_weak','parentheses_in_title','heuristic_bonus',
    'max_title_family_bm25','max_keywords_family_bm25','max_title_keywords_family_bm25','max_legacy_bm25',
    'min_title_family_rank','min_title_keywords_family_rank','min_legacy_rank','expanded_terms_count','expanded_overlap_ratio',
    *[f'{v.name}_rank' for v in VARIANTS], *[f'{v.name}_bm25' for v in VARIANTS]
]
DENSE_FEATURES = ['dense_title_cos','dense_keywords_cos','dense_title_keywords_cos','dense_short_text_cos','dense_max_cos','dense_mean_cos','dense_rank','dense_inv_rank']
CROSS_FEATURES = ['cross_score','cross_rank','cross_inv_rank','cross_score_minus_dense_max','has_cross_score']
MODEL_CONFIGS = [
    ('m31_lr003', dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, seed=13)),
    ('m63_lr002', dict(learning_rate=0.02, num_leaves=63, min_data_in_leaf=25, feature_fraction=0.85, bagging_fraction=0.9, bagging_freq=1, seed=23)),
    ('m15_lr005', dict(learning_rate=0.05, num_leaves=15, min_data_in_leaf=15, feature_fraction=0.95, bagging_fraction=0.85, bagging_freq=1, seed=37)),
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r'[^\w\s]+', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))  # no stopword removal


def bigrams(xs: list[str]) -> set[tuple[str, str]]:
    return set(zip(xs, xs[1:]))


def ordered_subsequence(needle: list[str], haystack: list[str]) -> bool:
    if not needle:
        return False
    j = 0
    for t in haystack:
        if t == needle[j]:
            j += 1
            if j == len(needle):
                return True
    return False


def read_queries(path: Path) -> list[tuple[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return [(r['QueryId'], r['Query']) for r in csv.DictReader(f)]


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open('r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            out[r['QueryId']][r['EntityId']] = int(r['Relevance'])
    return out


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(INDEX_PATH)
    con.execute('pragma temp_store=MEMORY')
    con.execute('pragma cache_size=-2000000')
    return con


def validate_index(con: sqlite3.Connection) -> None:
    tables = sorted({v.table for v in VARIANTS})
    missing = []
    for t in tables:
        if con.execute("select 1 from sqlite_master where type='table' and name=?", (t,)).fetchone() is None:
            missing.append(t)
    if missing:
        raise RuntimeError(f"Missing FTS tables: {missing}. Run build_index_v2.py first.")


def fts_query(query: str, operator: str) -> str:
    toks = tokenize(query)
    return '""' if not toks else f' {operator} '.join(f'"{t}"' for t in toks)


def run_variant(con: sqlite3.Connection, query: str, v: Variant, candidate_k: int) -> list[tuple[str, float, int]]:
    limit = max(50, int(candidate_k * v.per_variant_factor))
    expr = fts_query(query, v.operator)
    if v.weights is None:
        sql = f"select entity_id, bm25({v.table}) as s from {v.table} where {v.table} match ? order by s asc limit ?"
        params: tuple[Any, ...] = (expr, limit)
    else:
        tw, kw, xw = v.weights
        sql = f"select entity_id, bm25({v.table}, 0.0, ?, ?, ?) as s from {v.table} where {v.table} match ? order by s asc limit ?"
        params = (tw, kw, xw, expr, limit)
    try:
        rows = con.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(eid, -float(score), rank) for rank, (eid, score) in enumerate(rows, 1)]


def empty_stats(candidate_k: int) -> dict[str, float]:
    d = dict(rrf=0.0, best_bm25=0.0, best_rank=float(candidate_k+1), variant_hits=0.0, rank_sum=0.0, expanded_terms_count=0.0, expanded_overlap_ratio=0.0)
    for v in VARIANTS:
        d[f'{v.name}_rank'] = 0.0
        d[f'{v.name}_bm25'] = 0.0
    return d


def load_fields(con: sqlite3.Connection, ids: list[str]) -> dict[str, tuple[str, str, str]]:
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i+500]
        if not chunk:
            continue
        marks = ','.join('?' for _ in chunk)
        for eid, title, keywords, text in con.execute(f'select entity_id,title,keywords,text from entities where entity_id in ({marks})', chunk):
            out[eid] = (title or '', keywords or '', text or '')
    return out


def build_train_expansions(con, queries, qrels, max_terms=4):
    exp = {}
    for qid, q in tqdm(queries, desc='qrel expansion', unit='query'):
        rels = qrels.get(qid, {})
        fields = load_fields(con, [eid for eid, rel in rels.items() if rel > 0])
        q_terms = set(tokenize(q))
        c = Counter()
        for eid, rel in rels.items():
            if rel > 0 and eid in fields:
                title, keywords, _ = fields[eid]
                for tok in tokenize(title + ' ' + keywords):
                    if len(tok) > 2 and tok not in q_terms:
                        c[tok] += 2 if rel == 2 else 1
        exp[qid] = [t for t, _ in c.most_common(max_terms)]
    return exp


def build_global_expander(con, queries, qrels, top_terms=5):
    assoc: dict[str, Counter[str]] = defaultdict(Counter)
    for qid, q in tqdm(queries, desc='global expander', unit='query'):
        q_terms = set(tokenize(q))
        rels = qrels.get(qid, {})
        fields = load_fields(con, [eid for eid, rel in rels.items() if rel > 0])
        c = Counter()
        for eid, rel in rels.items():
            if rel > 0 and eid in fields:
                title, keywords, _ = fields[eid]
                for tok in tokenize(title + ' ' + keywords):
                    if len(tok) > 2 and tok not in q_terms:
                        c[tok] += 2 if rel == 2 else 1
        for qt in q_terms:
            for t, s in c.most_common(10):
                assoc[qt][t] += s
    return {k: [t for t, _ in c.most_common(top_terms)] for k, c in assoc.items()}


def expand_test_query(q: str, expander: dict[str, list[str]], max_terms=4) -> list[str]:
    q_terms = set(tokenize(q)); c = Counter()
    for qt in q_terms:
        for t in expander.get(qt, []):
            if t not in q_terms:
                c[t] += 1
    return [t for t, _ in c.most_common(max_terms)]


def collect_candidates(con, qid, q, candidate_k, rrf_k, train_exp=None, global_expander=None):
    cand: dict[str, dict[str, float]] = {}
    if train_exp and qid in train_exp:
        extra = train_exp[qid]
    elif global_expander:
        extra = expand_test_query(q, global_expander)
    else:
        extra = []
    expanded_q = ' '.join([q, *extra]).strip()
    orig_tokens, exp_tokens = set(tokenize(q)), set(tokenize(expanded_q))
    for v in VARIANTS:
        aq = expanded_q if v.query_mode == 'expanded' and extra else q
        for eid, bm25, rank in run_variant(con, aq, v, candidate_k):
            it = cand.setdefault(eid, empty_stats(candidate_k))
            it['rrf'] += v.weight / (rrf_k + rank)
            it['best_bm25'] = max(it['best_bm25'], bm25)
            it['best_rank'] = min(it['best_rank'], rank)
            it['variant_hits'] += 1.0
            it['rank_sum'] += rank
            it[f'{v.name}_rank'] = float(rank)
            it[f'{v.name}_bm25'] = bm25
            it['expanded_terms_count'] = float(len(extra))
            it['expanded_overlap_ratio'] = len(orig_tokens & exp_tokens) / (len(exp_tokens) or 1)
    return cand


def safe_min_pos(xs, default=0.0):
    ys = [x for x in xs if x > 0]
    return min(ys) if ys else default


def heuristic_bonus(q, fields):
    title, keywords, text = fields
    qn, qt, qs = normalize_text(q), tokenize(q), set(tokenize(q))
    if not qs:
        return 0.0
    tt, kt, xt = tokenize(title), tokenize(keywords), tokenize(text)
    ts, ks, xs = set(tt), set(kt), set(xt)
    tks = ts | ks
    denom = len(qs)
    tc, kc, xc, tkc = len(qs & ts)/denom, len(qs & ks)/denom, len(qs & xs)/denom, len(qs & tks)/denom
    qb = bigrams(qt); bd = len(qb) or 1
    tbc, kbc = len(qb & bigrams(tt))/bd, len(qb & bigrams(kt))/bd
    tn, kn = normalize_text(title), normalize_text(keywords)
    bonus = .42*tc + .22*kc + .04*xc + .24*tkc + .28*tbc + .12*kbc
    bonus += .55 if qn and qn in tn else 0
    bonus += .25 if qn and qn in kn else 0
    bonus += .35 if qs <= ts else 0
    bonus += .24 if qs <= tks else 0
    bonus += .16 if len(tt) <= 5 and tc >= .6 else 0
    bonus += .14 if len(tt) <= 8 and tc >= .8 else 0
    bonus -= .14 if len(tt) >= 18 and tc < .5 else 0
    bonus -= .10 if len(tt) >= 28 else 0
    return bonus


def lexical_features(q, stats, fields):
    title, keywords, text = fields
    qn, qt, qs = normalize_text(q), tokenize(q), set(tokenize(q)); denom = len(qs) or 1
    tt, kt, xt = tokenize(title), tokenize(keywords), tokenize(text)
    ts, ks, xs = set(tt), set(kt), set(xt); tks = ts | ks
    qb, tb, kb = bigrams(qt), bigrams(tt), bigrams(kt); tkb = tb | kb; bd = len(qb) or 1
    th, kh, xh = len(qs & ts), len(qs & ks), len(qs & xs)
    tc, kc, xc, tkc = th/denom, kh/denom, xh/denom, len(qs & tks)/denom
    tj = len(qs & ts)/(len(qs | ts) or 1); kj = len(qs & ks)/(len(qs | ks) or 1); tkj = len(qs & tks)/(len(qs | tks) or 1)
    tbc, kbc, tkbc = len(qb & tb)/bd, len(qb & kb)/bd, len(qb & tkb)/bd
    tn, kn, xn = normalize_text(title), normalize_text(keywords), normalize_text(text)
    br = stats['best_rank']; mr = stats['rank_sum']/max(stats['variant_hits'], 1.0)
    title_bm25 = [stats.get('title_and_bm25',0), stats.get('title_or_bm25',0)]
    kw_bm25 = [stats.get('keywords_and_bm25',0), stats.get('keywords_or_bm25',0)]
    tk_bm25 = [stats.get(x,0) for x in ['title_keywords_and_bm25','title_keywords_or_bm25','porter_title_keywords_or_bm25','expanded_title_keywords_or_bm25','expanded_porter_title_keywords_bm25']]
    legacy_bm25 = [stats.get(x,0) for x in ['legacy_balanced_bm25','legacy_title_heavy_bm25','legacy_title_keywords_bm25','legacy_keywords_heavy_bm25','legacy_porter_title_keywords_bm25','expanded_legacy_title_keywords_bm25']]
    title_ranks = [stats.get('title_and_rank',0), stats.get('title_or_rank',0)]
    tk_ranks = [stats.get(x,0) for x in ['title_keywords_and_rank','title_keywords_or_rank','porter_title_keywords_or_rank','expanded_title_keywords_or_rank','expanded_porter_title_keywords_rank']]
    legacy_ranks = [stats.get(x,0) for x in ['legacy_balanced_rank','legacy_title_heavy_rank','legacy_title_keywords_rank','legacy_keywords_heavy_rank','legacy_porter_title_keywords_rank','expanded_legacy_title_keywords_rank']]
    row = [
        stats['rrf'], br, 1/max(br,1), mr, 1/max(mr,1), stats['best_bm25'], math.log1p(max(stats['best_bm25'],0)),
        stats['variant_hits'], math.log1p(stats['variant_hits']), tc, kc, xc, tkc, tj, kj, tkj, float(th), float(kh), float(xh),
        tbc, kbc, tkbc, 1.0 if qb and qb & tb else 0.0, 1.0 if qb and qb & kb else 0.0,
        1.0 if qn and qn in tn else 0.0, 1.0 if qn and qn in kn else 0.0, 1.0 if qn and qn in xn else 0.0,
        1.0 if qn == tn else 0.0, 1.0 if qs and qs <= ts else 0.0, 1.0 if qs and qs <= ks else 0.0, 1.0 if qs and qs <= tks else 0.0,
        1.0 if qt and qt[0] in ts else 0.0, 1.0 if qt and qt[-1] in ts else 0.0, 1.0 if ordered_subsequence(qt, tt) else 0.0,
        float(len(tt)), float(len(kt)), float(len(xt)), float(len(title)), 1.0 if len(tt)<=5 and tc>=.6 else 0.0, 1.0 if len(tt)<=4 and tc>=.8 else 0.0,
        1.0 if len(tt)>=18 and tc<.5 else 0.0, 1.0 if '(' in title and ')' in title else 0.0, heuristic_bonus(q, fields),
        max(title_bm25), max(kw_bm25), max(tk_bm25), max(legacy_bm25), safe_min_pos(title_ranks), safe_min_pos(tk_ranks), safe_min_pos(legacy_ranks),
        stats['expanded_terms_count'], stats['expanded_overlap_ratio']
    ]
    row.extend(stats[f'{v.name}_rank'] for v in VARIANTS)
    row.extend(stats[f'{v.name}_bm25'] for v in VARIANTS)
    return [float(x) for x in row]


def load_dense_model(name, device=None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device) if device else SentenceTransformer(name)


def load_cross_model(name, device=None):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=device) if device else CrossEncoder(name)


def cache_path(kind, qid, q, ids, cfg):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(kind=kind, qid=qid, q=q, ids=hashlib.md5('\n'.join(ids).encode()).hexdigest(), cfg=cfg), sort_keys=True)
    return CACHE_DIR / f"{kind}_{hashlib.md5(raw.encode()).hexdigest()}.pkl"


def empty_dense(): return [0.0]*len(DENSE_FEATURES)
def empty_cross(): return [0.0]*len(CROSS_FEATURES)


def dense_features(model, q, ids, fields, batch_size):
    qemb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
    out = {}
    flat = []
    order = []
    for eid in ids:
        title, keywords, text = fields[eid]
        texts = [title, keywords, f'{title} {keywords}'.strip(), f'{title} {keywords} {text[:512]}'.strip()]
        flat.extend(texts); order.append(eid)
    embs = model.encode(flat, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size) if flat else []
    for i, eid in enumerate(order):
        scores = [float(np.dot(qemb, embs[4*i+j])) for j in range(4)]
        out[eid] = [*scores, max(scores), sum(scores)/4, 0.0, 0.0]
    for rank, eid in enumerate(sorted(out, key=lambda e: (-out[e][4], e)), 1):
        out[eid][6] = float(rank); out[eid][7] = 1/rank
    return out


def cross_features(model, q, ids, fields, dense_map, top_k, batch_size):
    selected = sorted(ids, key=lambda e: (-(dense_map.get(e, empty_dense())[4]), e))[:top_k] if dense_map else ids[:top_k]
    pairs = [(q, f"{fields[e][0]} [SEP] {fields[e][1]} [SEP] {fields[e][2][:512]}") for e in selected]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False) if pairs else []
    out = {eid: empty_cross() for eid in ids}
    for eid, s in zip(selected, scores):
        dense_max = dense_map.get(eid, empty_dense())[4] if dense_map else 0.0
        out[eid] = [float(s), 0.0, 0.0, float(s)-dense_max, 1.0]
    for rank, eid in enumerate(sorted(selected, key=lambda e: (-out[e][0], e)), 1):
        out[eid][1] = float(rank); out[eid][2] = 1/rank
    return out


def build_matrix(con, queries, qrels, args, train_exp=None, global_expander=None, dense_model=None, cross_model=None, augment=False):
    feature_names = list(LEXICAL_FEATURES) + (DENSE_FEATURES if args.use_dense else []) + (CROSS_FEATURES if args.use_cross_encoder else [])
    X, y, groups, keys, pools = [], [], [], [], {}
    for qid, q in tqdm(queries, desc='features', unit='query'):
        cand = collect_candidates(con, qid, q, args.candidate_k, args.rrf_k, train_exp, global_expander)
        if augment and qrels is not None:
            for eid in qrels.get(qid, {}): cand.setdefault(eid, empty_stats(args.candidate_k))
        ids = sorted(cand, key=lambda e: (-cand[e]['rrf'], cand[e]['best_rank'], e))[:args.candidate_k]
        fields = load_fields(con, ids)
        ids = [e for e in ids if e in fields]
        pools[qid] = ids
        dmap, cmap = {}, {}
        if args.use_dense:
            p = cache_path('dense', qid, q, ids, dict(model=args.dense_model))
            if p.exists() and not args.no_cache:
                dmap = pickle.load(open(p, 'rb'))
            else:
                dmap = dense_features(dense_model, q, ids, fields, args.dense_batch_size)
                if not args.no_cache: pickle.dump(dmap, open(p, 'wb'))
        if args.use_cross_encoder:
            p = cache_path('cross', qid, q, ids, dict(model=args.cross_encoder_model, top_k=args.cross_encoder_top_k))
            if p.exists() and not args.no_cache:
                cmap = pickle.load(open(p, 'rb'))
            else:
                cmap = cross_features(cross_model, q, ids, fields, dmap, args.cross_encoder_top_k, args.cross_batch_size)
                if not args.no_cache: pickle.dump(cmap, open(p, 'wb'))
        n = 0
        for eid in ids:
            row = lexical_features(q, cand[eid], fields[eid])
            if args.use_dense: row += dmap.get(eid, empty_dense())
            if args.use_cross_encoder: row += cmap.get(eid, empty_cross())
            X.append(row); keys.append((qid, eid)); n += 1
            if qrels is not None: y.append(qrels.get(qid, {}).get(eid, 0))
        groups.append(n)
    return np.asarray(X, np.float32), (np.asarray(y, np.int32) if qrels is not None else None), groups, keys, pools, feature_names


def dcg(rels, k): return sum((2**r-1)/math.log2(i+2) for i, r in enumerate(rels[:k]))

def ndcg(run, qrels, k=100):
    vals = []
    for qid, rels in qrels.items():
        ranking = run.get(qid, [])[:k]
        gains = [rels.get(e, 0) for e in ranking]
        ideal = sorted(rels.values(), reverse=True)[:k]
        idcg = dcg(ideal, k)
        vals.append(0.0 if idcg == 0 else dcg(gains, k)/idcg)
    return sum(vals)/len(vals) if vals else 0.0


def run_from_scores(keys, scores, top_k):
    byq = defaultdict(list)
    for (qid, eid), s in zip(keys, scores): byq[qid].append((eid, float(s)))
    return {qid: [e for e, _ in sorted(items, key=lambda x: (-x[1], x[0]))[:top_k]] for qid, items in byq.items()}


def oracle_recall(pools, qrels, k):
    rs, gs = [], []
    for qid, rels in qrels.items():
        top = set(pools.get(qid, [])[:k]); relids = set(rels)
        rs.append(len(top & relids)/len(relids))
        total = sum(2**r-1 for r in rels.values())
        hit = sum(2**r-1 for e, r in rels.items() if e in top)
        gs.append(hit/total if total else 0)
    return sum(rs)/len(rs), sum(gs)/len(gs)


def train_model(name, params, X, y, groups, features, Xv=None, yv=None, gv=None, rounds=300, early=100):
    base = dict(objective='lambdarank', metric='ndcg', eval_at=[100], label_gain=[0,1,3], num_threads=-1, verbosity=-1)
    base.update(params)
    dtrain = lgb.Dataset(X, label=y, group=groups, feature_name=features, free_raw_data=False)
    callbacks = [lgb.log_evaluation(50)]
    valids = names = None
    if Xv is not None:
        dval = lgb.Dataset(Xv, label=yv, group=gv, feature_name=features, reference=dtrain, free_raw_data=False)
        valids, names = [dval], ['valid']
        callbacks.append(lgb.early_stopping(early))
    print('training', name)
    return lgb.train(base, dtrain, valid_sets=valids, valid_names=names, num_boost_round=rounds, callbacks=callbacks)


def predict(models, X):
    preds = []
    for m in models:
        preds.append(m.predict(X, num_iteration=(m.best_iteration or None)))
    return np.mean(np.vstack(preds), axis=0)


def fallback_ids(con, limit=1000): return [r[0] for r in con.execute('select entity_id from entities order by entity_id limit ?', (limit,))]


def write_submission(path, run, queries, top_k, fillers):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['QueryId','EntityId'])
        for qid, _ in queries:
            ids = list(run.get(qid, [])[:top_k]); seen = set(ids)
            for e in fillers:
                if len(ids) >= top_k: break
                if e not in seen: ids.append(e); seen.add(e)
            for e in ids[:top_k]: w.writerow([qid, e])


def load_optional_models(args):
    dense = cross = None
    if args.use_dense:
        print('loading dense', args.dense_model); dense = load_dense_model(args.dense_model, args.device)
    if args.use_cross_encoder:
        print('loading cross', args.cross_encoder_model); cross = load_cross_model(args.cross_encoder_model, args.device)
    return dense, cross


def split_queries(qs): return [q for i,q in enumerate(qs) if i%5 != 0], [q for i,q in enumerate(qs) if i%5 == 0]


def eval_mode(args, con, queries, qrels):
    dense, cross = load_optional_models(args)
    train_exp = build_train_expansions(con, queries, qrels, args.expansion_terms)
    glob = build_global_expander(con, queries, qrels) if args.use_global_expander else None
    fit, valid = split_queries(queries)
    X, y, g, _, _, feats = build_matrix(con, fit, qrels, args, train_exp, glob, dense, cross, args.augment_qrels)
    Xv, yv, gv, keys, pools, _ = build_matrix(con, valid, qrels, args, train_exp, glob, dense, cross, False)
    configs = MODEL_CONFIGS if args.ensemble else [MODEL_CONFIGS[0]]
    models = [train_model(n,p,X,y,g,feats,Xv,yv,gv,args.num_boost_round,args.early_stopping_rounds) for n,p in configs]
    scores = predict(models, Xv)
    run = run_from_scores(keys, scores, args.top_k)
    base = {qid: ids[:args.top_k] for qid, ids in pools.items()}
    vq = {qid: qrels[qid] for qid,_ in valid if qid in qrels}
    r100, g100 = oracle_recall(pools, vq, args.top_k)
    rpool, gpool = oracle_recall(pools, vq, args.candidate_k)
    print(f'features: train={X.shape} valid={Xv.shape} n_features={len(feats)}')
    print(f'rrf nDCG@{args.top_k}: {ndcg(base, vq, args.top_k):.6f}')
    print(f'ltr nDCG@{args.top_k}: {ndcg(run, vq, args.top_k):.6f}')
    print(f'pool recall@{args.top_k}: {r100:.6f} gain_recall@{args.top_k}: {g100:.6f}')
    print(f'pool recall@{args.candidate_k}: {rpool:.6f} gain_recall@{args.candidate_k}: {gpool:.6f}')
    print('best_iterations:', [m.best_iteration for m in models])


def train_mode(args, con, queries, qrels):
    dense, cross = load_optional_models(args)
    train_exp = build_train_expansions(con, queries, qrels, args.expansion_terms)
    glob = build_global_expander(con, queries, qrels) if args.use_global_expander else None
    X, y, g, _, _, feats = build_matrix(con, queries, qrels, args, train_exp, glob, dense, cross, args.augment_qrels)
    configs = MODEL_CONFIGS if args.ensemble else [MODEL_CONFIGS[0]]
    models = [train_model(n,p,X,y,g,feats,rounds=args.num_boost_round,early=args.early_stopping_rounds) for n,p in configs]
    payload = dict(models=models, feature_names=feats, candidate_k=args.candidate_k, rrf_k=args.rrf_k, top_k=args.top_k,
                   use_dense=args.use_dense, dense_model=args.dense_model, dense_batch_size=args.dense_batch_size,
                   use_cross_encoder=args.use_cross_encoder, cross_encoder_model=args.cross_encoder_model,
                   cross_encoder_top_k=args.cross_encoder_top_k, cross_batch_size=args.cross_batch_size,
                   ensemble=args.ensemble, augment_qrels=args.augment_qrels, expansion_terms=args.expansion_terms,
                   use_global_expander=args.use_global_expander, train_expansions=train_exp, global_expander=glob,
                   variants=[asdict(v) for v in VARIANTS], num_boost_round=args.num_boost_round)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    pickle.dump(payload, open(MODEL_PATH, 'wb'))
    print(f'trained {len(models)} model(s), X={X.shape}, saved={MODEL_PATH}')


def submission_mode(args, con):
    payload = pickle.load(open(MODEL_PATH, 'rb'))
    args.candidate_k = payload['candidate_k']; args.rrf_k = payload['rrf_k']; args.top_k = payload['top_k']
    args.use_dense = payload['use_dense']; args.dense_model = payload['dense_model']; args.dense_batch_size = payload['dense_batch_size']
    args.use_cross_encoder = payload['use_cross_encoder']; args.cross_encoder_model = payload['cross_encoder_model']; args.cross_encoder_top_k = payload['cross_encoder_top_k']; args.cross_batch_size = payload['cross_batch_size']
    args.ensemble = payload['ensemble']; args.augment_qrels = payload['augment_qrels']; args.expansion_terms = payload['expansion_terms']; args.use_global_expander = payload['use_global_expander']
    dense, cross = load_optional_models(args)
    queries = read_queries(DATA_DIR/'test_queries.csv')
    X, _, _, keys, _, feats = build_matrix(con, queries, None, args, payload.get('train_expansions'), payload.get('global_expander'), dense, cross, False)
    if feats != payload['feature_names']: raise RuntimeError('Feature mismatch')
    scores = predict(payload['models'], X)
    run = run_from_scores(keys, scores, args.top_k)
    write_submission(args.out, run, queries, args.top_k, fallback_ids(con))
    print('wrote', args.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['eval','train','submission'], default='eval')
    p.add_argument('--candidate-k', type=int, default=1000); p.add_argument('--rrf-k', type=int, default=60); p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--out', type=Path, default=ARTIFACTS_DIR/'submission_ltr_v3.csv')
    p.add_argument('--num-boost-round', type=int, default=300); p.add_argument('--early-stopping-rounds', type=int, default=100)
    p.add_argument('--augment-qrels', action='store_true'); p.add_argument('--ensemble', action='store_true')
    p.add_argument('--expansion-terms', type=int, default=4); p.add_argument('--use-global-expander', action='store_true')
    p.add_argument('--use-dense', action='store_true'); p.add_argument('--dense-model', default='sentence-transformers/all-MiniLM-L6-v2'); p.add_argument('--dense-batch-size', type=int, default=256)
    p.add_argument('--use-cross-encoder', action='store_true'); p.add_argument('--cross-encoder-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2'); p.add_argument('--cross-encoder-top-k', type=int, default=200); p.add_argument('--cross-batch-size', type=int, default=128)
    p.add_argument('--device', default=None); p.add_argument('--no-cache', action='store_true')
    args = p.parse_args()
    con = connect(); validate_index(con)
    try:
        if args.mode == 'submission':
            submission_mode(args, con); return
        queries = read_queries(DATA_DIR/'train_queries.csv'); qrels = read_qrels(DATA_DIR/'train_qrels.csv')
        if args.mode == 'eval': eval_mode(args, con, queries, qrels)
        elif args.mode == 'train': train_mode(args, con, queries, qrels)
    finally:
        con.close()

if __name__ == '__main__': main()
