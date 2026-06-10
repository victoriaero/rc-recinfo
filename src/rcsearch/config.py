from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_PATHS = {
    "data_dir": "data",
    "artifact_dir": "artifacts",
    "index_path": "artifacts/indexes/entities_fts.sqlite",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    config.setdefault("paths", {})
    for key, value in DEFAULT_PATHS.items():
        config["paths"].setdefault(key, value)
    config["_config_path"] = str(config_path)
    return config


def ensure_artifact_dirs(config: dict[str, Any]) -> None:
    artifact_dir = Path(config["paths"]["artifact_dir"])
    for name in ["indexes", "runs", "submissions", "metadata", "features", "models", "grid"]:
        (artifact_dir / name).mkdir(parents=True, exist_ok=True)
