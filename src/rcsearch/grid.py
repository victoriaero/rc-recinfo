from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from .data import read_qrels, read_queries
from .eval import ndcg_at_k, write_grid_results
from .metadata import write_metadata
from .retrieve import retrieve_rows, write_run


def grid_search(config: dict[str, Any]) -> dict[str, float]:
    data_dir = Path(config["paths"]["data_dir"])
    queries = read_queries(data_dir / "train_queries.csv")
    qrels = read_qrels(data_dir / "train_qrels.csv")
    grid = config.get("grid", {})
    title_values = grid.get("title", [1, 2, 3, 5, 8])
    keyword_values = grid.get("keywords", [1, 2, 3, 5])
    text_values = grid.get("text", [0.5, 1, 2])
    top_k = int(config.get("retrieval", {}).get("top_k", 100))
    run_name = config.get("run", {}).get("name", "grid")

    results = []
    best: dict[str, float] = {"title": 1.0, "keywords": 1.0, "text": 1.0, "ndcg_at_100": -1.0}
    for title, keywords, text in product(title_values, keyword_values, text_values):
        trial = dict(config)
        trial["retrieval"] = dict(config.get("retrieval", {}))
        trial["retrieval"]["mode"] = "fielded"
        trial["retrieval"]["weights"] = {"title": title, "keywords": keywords, "text": text}
        rows = retrieve_rows(trial, queries, top_k=top_k)
        score = ndcg_at_k(rows, qrels, k=100)
        result = {"title": float(title), "keywords": float(keywords), "text": float(text), "ndcg_at_100": score}
        results.append(result)
        if score > best["ndcg_at_100"]:
            best = result

    artifact_dir = Path(config["paths"]["artifact_dir"])
    grid_path = artifact_dir / "grid" / f"{run_name}.csv"
    write_grid_results(grid_path, results)

    best_config = dict(config)
    best_config["retrieval"] = dict(config.get("retrieval", {}))
    best_config["retrieval"]["mode"] = "fielded"
    best_config["retrieval"]["weights"] = {
        "title": best["title"],
        "keywords": best["keywords"],
        "text": best["text"],
    }
    train_run = artifact_dir / "runs" / f"{run_name}_train.csv"
    write_run(train_run, retrieve_rows(best_config, queries, top_k=top_k))

    write_metadata(
        artifact_dir / "metadata" / f"{run_name}_grid.json",
        {
            "command": "grid-search",
            "config_path": config.get("_config_path"),
            "grid_path": str(grid_path),
            "best": best,
            "top_k": top_k,
            "query_count": len(queries),
        },
    )
    return best
