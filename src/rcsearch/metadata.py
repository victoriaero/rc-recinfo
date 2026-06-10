from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with output.open("w", encoding="utf-8") as fh:
        json.dump(enriched, fh, indent=2, ensure_ascii=False, sort_keys=True)
