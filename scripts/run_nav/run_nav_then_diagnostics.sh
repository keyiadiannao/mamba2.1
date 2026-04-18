#!/usr/bin/env bash
# Run one or more navigation batch configs, then analyze_evidence_saturation + audit_accept_gate
# for each batch_id (parsed from __SSGS_BATCH_ID__= line printed by run_navigation_batch.py).
#
# Usage (from repo root):
#   bash scripts/run_nav/run_nav_then_diagnostics.sh \
#     configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_rule.example.json \
#     configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_learned_root_blend05.example.json
#
# Optional: extra args for every navigation run (whitespace-separated):
#   NAV_BATCH_EXTRA_ARGS='--max-samples 10' bash scripts/run_nav/run_nav_then_diagnostics.sh configs/...
#
# Optional: registry path (default outputs/reports/run_registry.jsonl):
#   REGISTRY_JSONL=outputs/reports/run_registry.jsonl bash scripts/run_nav/run_nav_then_diagnostics.sh ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: NAV_BATCH_EXTRA_ARGS='--max-samples 10' bash $0 <config.json> [config2.json ...]" >&2
  exit 2
fi

REGISTRY_JSONL="${REGISTRY_JSONL:-outputs/reports/run_registry.jsonl}"
EXTRA=()
if [[ -n "${NAV_BATCH_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "${NAV_BATCH_EXTRA_ARGS}"
fi

for cfg in "$@"; do
  echo "========== navigation batch: ${cfg} =========="
  mapfile -t lines < <(python scripts/run_nav/run_navigation_batch.py --config "$cfg" "${EXTRA[@]}" 2>&1)
  printf '%s\n' "${lines[@]}"
  bid=""
  for line in "${lines[@]}"; do
    case "$line" in
      __SSGS_BATCH_ID__=*) bid="${line#__SSGS_BATCH_ID__=}" ;;
    esac
  done
  if [[ -z "$bid" ]]; then
    echo "error: could not parse __SSGS_BATCH_ID__= from output for ${cfg}" >&2
    exit 1
  fi
  echo "---------- diagnostics: ${bid} ----------"
  python scripts/diagnostics/analyze_evidence_saturation.py \
    --registry-jsonl "$REGISTRY_JSONL" \
    --batch-id "$bid" \
    --out-json "outputs/reports/evidence_saturation_${bid}.json"
  python scripts/diagnostics/audit_accept_gate.py \
    --registry-jsonl "$REGISTRY_JSONL" \
    --batch-id "$bid"
done
