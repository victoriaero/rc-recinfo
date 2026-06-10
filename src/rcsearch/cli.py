from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ensure_artifact_dirs, load_config
from .data import read_queries
from .eval import evaluate_run
from .export import export_submission
from .grid import grid_search
from .index import index_corpus
from .metadata import write_metadata
from .rerank import rerank, train_reranker
from .retrieve import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(prog="rcsearch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index")
    build.add_argument("--config", required=True)
    build.add_argument("--reset", action="store_true")
    build.add_argument("--limit", type=int)

    ret = subparsers.add_parser("retrieve")
    ret.add_argument("--config", required=True)
    ret.add_argument("--split", choices=["train", "test"], default="test")
    ret.add_argument("--top-k", type=int)
    ret.add_argument("--out")

    ev = subparsers.add_parser("evaluate")
    ev.add_argument("--run", required=True)
    ev.add_argument("--qrels", default="data/train_qrels.csv")
    ev.add_argument("--k", type=int, default=100)

    exp = subparsers.add_parser("export")
    exp.add_argument("--run", required=True)
    exp.add_argument("--out", required=True)
    exp.add_argument("--test-queries", default="data/test_queries.csv")
    exp.add_argument("--top-k", type=int, default=100)

    grid = subparsers.add_parser("grid-search")
    grid.add_argument("--config", required=True)

    train = subparsers.add_parser("train-reranker")
    train.add_argument("--config", required=True)

    rer = subparsers.add_parser("rerank")
    rer.add_argument("--config", required=True)

    args = parser.parse_args()

    config = None
    if hasattr(args, "config"):
        config = load_config(args.config)
        ensure_artifact_dirs(config)

    if args.command == "build-index":
        count = index_corpus(config, reset=args.reset, limit=args.limit)
        print(f"Indexed {count} entities this run.")
    elif args.command == "retrieve":
        path = retrieve(config, split=args.split, top_k=args.top_k, output=args.out)
        print(path)
    elif args.command == "evaluate":
        score = evaluate_run(args.run, args.qrels, k=args.k)
        print(json.dumps({"ndcg_at_k": score, "k": args.k}, indent=2))
    elif args.command == "export":
        path = export_submission(args.run, args.out, args.test_queries, top_k=args.top_k)
        write_metadata(
            Path("artifacts") / "metadata" / f"{Path(args.out).stem}_export.json",
            {
                "command": "export",
                "run_path": args.run,
                "submission_path": str(path),
                "test_queries": args.test_queries,
                "top_k": args.top_k,
                "query_count": len(read_queries(args.test_queries)),
            },
        )
        print(path)
    elif args.command == "grid-search":
        best = grid_search(config)
        print(json.dumps(best, indent=2))
    elif args.command == "train-reranker":
        print(train_reranker(config))
    elif args.command == "rerank":
        print(rerank(config))
