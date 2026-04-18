#!/usr/bin/env python3
"""Audit Controller accept gate vs gold leaves (offline, read run_payload.json).

Quantifies, for samples with gold annotation:
- gold never visited (navigation / ordering issue)
- gold visited but not in accept_evidence (accept / threshold / cap / duplicate)

Usage (from repo root):

  python scripts/diagnostics/audit_accept_gate.py \\
    --registry-jsonl outputs/reports/run_registry.jsonl \\
    --batch-id 'end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_20260417_154358Z' \\
    --out-json outputs/reports/accept_gate_audit_rule_p0.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.diagnostics.accept_gate_audit import audit_payload, summarize


def _load_aes_module():
    path = _REPO_ROOT / "scripts" / "diagnostics" / "analyze_evidence_saturation.py"
    spec = importlib.util.spec_from_file_location("_aes_audit_shim", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load analyze_evidence_saturation")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry-jsonl", type=str, required=True)
    p.add_argument("--batch-id", type=str, default=None)
    p.add_argument("--batch-id-substring", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--out-json", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    aes = _load_aes_module()

    if args.batch_id and args.batch_id_substring:
        print("Use only one of --batch-id or --batch-id-substring", file=sys.stderr)
        return 2

    reg = Path(args.registry_jsonl)
    if not reg.is_absolute():
        reg = root / reg

    paths, reg_stats = aes._iter_payload_paths_from_registry(
        root,
        reg,
        args.batch_id,
        args.batch_id_substring,
        args.limit,
    )
    if not paths:
        print("No payloads matched.", file=sys.stderr)
        print(json.dumps({"registry_debug": reg_stats}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2

    per_sample: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            payload = aes._load_json(path)
            row = audit_payload(payload)
            row["source_path"] = str(path)
            per_sample.append(row)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            errors.append(f"{path}: {exc}")

    summary = summarize(per_sample)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    report = {
        "summary": summary,
        "per_sample": per_sample,
        "errors": errors,
        "inputs": {
            "root": str(root),
            "registry_jsonl": str(reg),
            "batch_id": args.batch_id,
            "batch_id_substring": args.batch_id_substring,
            "payload_count": len(paths),
            "registry_debug": reg_stats,
        },
    }

    if args.out_json:
        out_path = Path(args.out_json)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote full report to {out_path}", file=sys.stderr)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
