from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import ltr_pipeline_v3 as base


QWEN_RERANKER_4B = "Qwen/Qwen3-Reranker-4B"
DEFAULT_PROMPT = (
    "Given an entity-search query, judge whether the entity is relevant to the "
    "query. The document describes a Wikipedia-like entity."
)
DEFAULT_DOC_CHARS = 512
DEFAULT_CROSS_TOP_K = 200
DEFAULT_CROSS_BATCH_SIZE = 8


def has_arg(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def add_default_bool(argv: list[str], name: str) -> None:
    if not has_arg(argv, name):
        argv.append(name)


def add_default_value(argv: list[str], name: str, value: str | int | Path) -> None:
    if not has_arg(argv, name):
        argv.extend([name, str(value)])


def get_arg_value(argv: list[str], name: str, default: str) -> str:
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return default


def parse_qwen_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qwen-no-dense", action="store_true")
    parser.add_argument("--qwen-no-ensemble", action="store_true")
    parser.add_argument("--qwen-no-global-expander", action="store_true")
    parser.add_argument("--qwen-doc-chars", type=int, default=DEFAULT_DOC_CHARS)
    parser.add_argument("--qwen-max-length", type=int, default=1024)
    parser.add_argument("--qwen-prompt", default=DEFAULT_PROMPT)
    return parser.parse_known_args(argv)


def configure_base_paths() -> None:
    base.MODEL_PATH = base.ARTIFACTS_DIR / "ltr_model_v3_qwen4b.pkl"


def load_qwen_cross_model(name: str, device: str | None = None):
    from sentence_transformers import CrossEncoder

    model_kwargs = {"torch_dtype": "auto"}
    return CrossEncoder(
        name,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        max_length=QWEN_MAX_LENGTH,
    )


def format_entity_doc(fields: tuple[str, str, str], doc_chars: int) -> str:
    title, keywords, text = fields
    return (
        f"Entity: {title}\n"
        f"Keywords: {keywords}\n"
        f"Description: {text[:doc_chars]}"
    ).strip()


def qwen_cross_features(model, q, ids, fields, dense_map, top_k, batch_size):
    if dense_map:
        selected = sorted(
            ids,
            key=lambda e: (-(dense_map.get(e, base.empty_dense())[4]), e),
        )[:top_k]
    else:
        selected = ids[:top_k]

    pairs = [(q, format_entity_doc(fields[e], QWEN_DOC_CHARS)) for e in selected]
    out = {eid: base.empty_cross() for eid in ids}
    if not pairs:
        return out

    scores = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        prompt=QWEN_PROMPT,
    )
    scores = np.asarray(scores).reshape(-1)

    for eid, s in zip(selected, scores):
        dense_max = dense_map.get(eid, base.empty_dense())[4] if dense_map else 0.0
        out[eid] = [float(s), 0.0, 0.0, float(s) - dense_max, 1.0]

    for rank, eid in enumerate(sorted(selected, key=lambda e: (-out[e][0], e)), 1):
        out[eid][1] = float(rank)
        out[eid][2] = 1 / rank
    return out


def configure_argv(argv: list[str], qwen_args: argparse.Namespace) -> list[str]:
    configured = list(argv)
    mode = get_arg_value(configured, "--mode", "eval")

    add_default_bool(configured, "--use-cross-encoder")
    add_default_value(configured, "--cross-encoder-model", QWEN_RERANKER_4B)
    add_default_value(configured, "--cross-encoder-top-k", DEFAULT_CROSS_TOP_K)
    add_default_value(configured, "--cross-batch-size", DEFAULT_CROSS_BATCH_SIZE)

    if not qwen_args.qwen_no_dense:
        add_default_bool(configured, "--use-dense")
    if not qwen_args.qwen_no_ensemble:
        add_default_bool(configured, "--ensemble")
    if not qwen_args.qwen_no_global_expander:
        add_default_bool(configured, "--use-global-expander")

    if mode == "submission":
        add_default_value(
            configured,
            "--out",
            base.ARTIFACTS_DIR / "submission_ltr_v3_qwen4b.csv",
        )

    return configured


def main() -> None:
    global QWEN_DOC_CHARS, QWEN_MAX_LENGTH, QWEN_PROMPT

    qwen_args, remaining = parse_qwen_args(sys.argv[1:])
    QWEN_DOC_CHARS = qwen_args.qwen_doc_chars
    QWEN_MAX_LENGTH = qwen_args.qwen_max_length
    QWEN_PROMPT = qwen_args.qwen_prompt

    configure_base_paths()
    base.load_cross_model = load_qwen_cross_model
    base.cross_features = qwen_cross_features

    sys.argv = [sys.argv[0], *configure_argv(remaining, qwen_args)]
    base.main()


if __name__ == "__main__":
    main()
