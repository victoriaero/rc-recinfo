from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

from .data import read_qrels, read_queries
from .features import FEATURE_NAMES, generate_features
from .metadata import write_metadata
from .retrieve import RunRow, write_run


def train_reranker(config: dict[str, Any]) -> Path:
    artifact_dir = Path(config["paths"]["artifact_dir"])
    run_name = config.get("run", {}).get("name", "submission_05")
    model_lib = import_ranker()
    if model_lib is None:
        write_metadata(
            artifact_dir / "metadata" / f"{run_name}_train_reranker_blocked.json",
            {
                "command": "train-reranker",
                "status": "blocked",
                "reason": "Neither lightgbm nor xgboost is importable in this Python 3.14 environment.",
                "preferred_libraries": ["lightgbm", "xgboost"],
            },
        )
        raise RuntimeError("Neither lightgbm nor xgboost is importable in this Python 3.14 environment.")

    train_run = artifact_dir / "runs" / f"{run_name}_train_candidates.csv"
    query_map = {query.query_id: query.text for query in read_queries(Path(config["paths"]["data_dir"]) / "train_queries.csv")}
    feature_path = artifact_dir / "features" / f"{run_name}_train.csv"
    generate_features(config, train_run, feature_path, query_map)

    qrels = read_qrels(Path(config["paths"]["data_dir"]) / "train_qrels.csv")
    X, y, groups = load_training_matrix(feature_path, qrels)
    if model_lib == "lightgbm":
        import lightgbm as lgb

        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05)
    else:
        import xgboost as xgb

        model = xgb.XGBRanker(objective="rank:ndcg", n_estimators=200, learning_rate=0.05)
    model.fit(X, y, group=groups)
    model_path = artifact_dir / "models" / f"{run_name}.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(model, fh)
    write_metadata(
        artifact_dir / "metadata" / f"{run_name}_train_reranker.json",
        {"command": "train-reranker", "library": model_lib, "model_path": str(model_path), "feature_path": str(feature_path)},
    )
    return model_path


def rerank(config: dict[str, Any]) -> Path:
    artifact_dir = Path(config["paths"]["artifact_dir"])
    run_name = config.get("run", {}).get("name", "submission_05")
    model_path = artifact_dir / "models" / f"{run_name}.pkl"
    test_run = artifact_dir / "runs" / f"{run_name}_test_candidates.csv"
    query_map = {query.query_id: query.text for query in read_queries(Path(config["paths"]["data_dir"]) / "test_queries.csv")}
    feature_path = artifact_dir / "features" / f"{run_name}_test.csv"
    generate_features(config, test_run, feature_path, query_map)
    with model_path.open("rb") as fh:
        model = pickle.load(fh)
    rows = load_prediction_rows(feature_path, model)
    output = artifact_dir / "runs" / f"{run_name}_test.csv"
    write_run(output, rows)
    write_metadata(
        artifact_dir / "metadata" / f"{run_name}_rerank.json",
        {"command": "rerank", "model_path": str(model_path), "feature_path": str(feature_path), "run_path": str(output)},
    )
    return output


def import_ranker() -> str | None:
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    try:
        import xgboost  # noqa: F401

        return "xgboost"
    except ImportError:
        return None


def load_training_matrix(path: Path, qrels: dict[str, dict[str, int]]) -> tuple[list[list[float]], list[int], list[int]]:
    X: list[list[float]] = []
    y: list[int] = []
    groups: list[int] = []
    current_qid = None
    current_count = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if current_qid is None:
                current_qid = row["QueryId"]
            if row["QueryId"] != current_qid:
                groups.append(current_count)
                current_qid = row["QueryId"]
                current_count = 0
            X.append([float(row[name]) for name in FEATURE_NAMES])
            y.append(qrels.get(row["QueryId"], {}).get(row["EntityId"], 0))
            current_count += 1
    if current_count:
        groups.append(current_count)
    return X, y, groups


def load_prediction_rows(path: Path, model: Any) -> list[RunRow]:
    candidates: dict[str, list[tuple[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        features = []
        keys = []
        for row in reader:
            keys.append((row["QueryId"], row["EntityId"]))
            features.append([float(row[name]) for name in FEATURE_NAMES])
    scores = model.predict(features)
    for (query_id, entity_id), score in zip(keys, scores):
        candidates.setdefault(query_id, []).append((entity_id, float(score)))
    output: list[RunRow] = []
    for query_id, hits in candidates.items():
        for rank, (entity_id, score) in enumerate(sorted(hits, key=lambda item: item[1], reverse=True), start=1):
            output.append(RunRow(query_id, entity_id, rank, score))
    return output
