#!/usr/bin/env bash
# P0-A / 实验记录 §9.12：对 B 链 500 的 rule（overlap_k4）与 cosine_probe 跑证据饱和度 + 生成器 context 金指标。
# 用法（在仓库根目录）：
#   bash scripts/diagnostics/run_9p12_rule_cosine_context_gold_saturation.sh
# 可选环境变量：
#   REG=outputs/reports/run_registry.jsonl   注册表路径（相对仓库根或绝对路径）
#   ROOT=/path/to/mamba2.1                    覆盖仓库根（默认为本脚本上两级目录）
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
REG="${REG:-outputs/reports/run_registry.jsonl}"

RULE_BATCH="${RULE_BATCH:-end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k4_20260416_140204Z}"
COSINE_BATCH="${COSINE_BATCH:-end_to_end_real_corpus_370m_qwen7b_cosine_probe_20260416_153243Z}"

mkdir -p outputs/reports

echo "==> Registry: $REG (cwd=$ROOT)"
if [[ ! -f "$REG" ]]; then
  echo "ERROR: registry not found: $REG" >&2
  exit 1
fi

run_one() {
  local batch_id="$1"
  local slug="$2"
  local out_json="outputs/reports/evidence_saturation_9p12_${slug}_ctxgold.json"
  local out_csv="outputs/reports/evidence_saturation_9p12_${slug}_ctxgold_per_sample.csv"
  echo ""
  echo "==> batch_id=$batch_id"
  python scripts/diagnostics/analyze_evidence_saturation.py \
    --root "$ROOT" \
    --registry-jsonl "$REG" \
    --batch-id "$batch_id" \
    --with-context-gold-metrics \
    --out-json "$out_json" \
    --per-sample-csv "$out_csv"
}

run_one "$RULE_BATCH" "rule_overlap_k4"
run_one "$COSINE_BATCH" "cosine_probe"

echo ""
echo "Done. JSON/CSV under outputs/reports/evidence_saturation_9p12_*_ctxgold*"
