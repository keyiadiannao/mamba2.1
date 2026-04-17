#!/usr/bin/env bash
# Smoke sweep: same manifest + same router checkpoint; only alpha and batch_id_prefix change.
# Training data / export are NOT modified (--max-samples only truncates which questions run).
#
# Usage (repo root):
#   chmod +x scripts/run_nav/run_blend_alpha_smoke.sh
#   export NAV_SMOKE_N=50
#   ./scripts/run_nav/run_blend_alpha_smoke.sh
#
# Optional: REPO_ROOT, PYTHON

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export REPO_ROOT
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python3}"
N="${NAV_SMOKE_N:-50}"
export NAV_SMOKE_N="$N"
export STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%SZ)}"

TMPDIR="$REPO_ROOT/outputs/reports/tmp_blend_smoke_configs"
mkdir -p "$TMPDIR"

base_learned="$REPO_ROOT/configs/experiment/navigation_batch_real_corpus_learned_root_v2.example.json"
base_rule="$REPO_ROOT/configs/experiment/navigation_batch_real_corpus_rule_frozen_heuristic_cos07_probe1_e8_pool20.example.json"

if [[ ! -f "$base_learned" ]] || [[ ! -f "$base_rule" ]]; then
  echo "Missing example configs under configs/experiment/" >&2
  exit 1
fi

echo "==> Rule baseline (smoke N=$N)"
RULE_CFG="$TMPDIR/nav_smoke_${N}_rule_${STAMP}.json"
export RULE_CFG
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
n = int(os.environ["NAV_SMOKE_N"])
stamp = os.environ["STAMP"]
repo = os.environ["REPO_ROOT"]
src = Path(repo) / "configs/experiment/navigation_batch_real_corpus_rule_frozen_heuristic_cos07_probe1_e8_pool20.example.json"
out = Path(os.environ["RULE_CFG"])
cfg = json.loads(src.read_text(encoding="utf-8"))
pfx = f"nav_smoke{n}_rule_frozen_{stamp}"
cfg["batch_id_prefix"] = pfx
cfg["run_id_prefix"] = pfx
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
PY
"$PYTHON" scripts/run_nav/run_navigation_batch.py --config "$RULE_CFG" --max-samples "$N"

for alpha in 0.0 0.25 0.5; do
  tag="${alpha//./_}"
  echo "==> Learned root blend alpha=$alpha (smoke N=$N)"
  BLEND_CFG="$TMPDIR/nav_smoke${N}_learned_blend_a${tag}_${STAMP}.json"
  export BLEND_CFG
  export BLEND_ALPHA="$alpha"
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
n = int(os.environ["NAV_SMOKE_N"])
stamp = os.environ["STAMP"]
alpha = float(os.environ["BLEND_ALPHA"])
repo = os.environ["REPO_ROOT"]
src = Path(repo) / "configs/experiment/navigation_batch_real_corpus_learned_root_v2.example.json"
out = Path(os.environ["BLEND_CFG"])
cfg = json.loads(src.read_text(encoding="utf-8"))
cfg["learned_root_blend_alpha"] = alpha
tag = str(alpha).replace(".", "_")
pfx = f"nav_smoke{n}_learned_blend_a{tag}_{stamp}"
cfg["batch_id_prefix"] = pfx
cfg["run_id_prefix"] = pfx
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
PY
  "$PYTHON" scripts/run_nav/run_navigation_batch.py --config "$BLEND_CFG" --max-samples "$N"
done

echo "Done. Summarize with: python scripts/diagnostics/analyze_evidence_saturation.py --registry-jsonl outputs/reports/run_registry.jsonl --batch-id '<id>'"
