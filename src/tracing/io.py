from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def make_run_id(prefix: str = "run") -> str:
    """UTC timestamp suffix for stable ordering across hosts and containers."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}Z"


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path


def write_run_payload(
    output_dir: str | Path,
    payload: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    final_run_id = run_id or make_run_id()
    run_dir = Path(output_dir) / final_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return write_json(run_dir / "run_payload.json", payload)
