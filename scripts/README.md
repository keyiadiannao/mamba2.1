# `scripts/` 脚本目录约定

- `build_tree/`: 构建树索引、摘要树、预处理脚本
- `run_nav/`: 导航阶段运行脚本
- `run_eval/`: 评测、汇总、导出报告脚本
- `utils/`: 数据搬运、检查、转换等辅助脚本

建议把“可重复执行的流程”写成脚本放这里，不要把一次性命令长期留在终端历史里。

`diagnostics/`（过程指标，**不启动评测**）：

- **`analyze_evidence_saturation.py`**：按 `run_registry.jsonl` 的 **`batch_id`**（或 `glob`）聚合 `run_payload.json`；加 **`--with-context-gold-metrics`** 时加载每条的 **`tree_path`**，统计生成器 context 与金叶子文本对齐情况。  
- **§9.12 双批一键跑（Linux 服务器，仓库根目录）**：

```bash
git pull origin main
bash scripts/diagnostics/run_9p12_rule_cosine_context_gold_saturation.sh
```

若注册表不在默认路径：

```bash
REG=/path/to/run_registry.jsonl bash scripts/diagnostics/run_9p12_rule_cosine_context_gold_saturation.sh
```

等价手写两条（便于改 `batch_id`）：

```bash
python scripts/diagnostics/analyze_evidence_saturation.py \
  --registry-jsonl outputs/reports/run_registry.jsonl \
  --batch-id 'end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k4_20260416_140204Z' \
  --with-context-gold-metrics \
  --out-json outputs/reports/evidence_saturation_9p12_rule_overlap_k4_ctxgold.json \
  --per-sample-csv outputs/reports/evidence_saturation_9p12_rule_overlap_k4_ctxgold_per_sample.csv

python scripts/diagnostics/analyze_evidence_saturation.py \
  --registry-jsonl outputs/reports/run_registry.jsonl \
  --batch-id 'end_to_end_real_corpus_370m_qwen7b_cosine_probe_20260416_153243Z' \
  --with-context-gold-metrics \
  --out-json outputs/reports/evidence_saturation_9p12_cosine_probe_ctxgold.json \
  --per-sample-csv outputs/reports/evidence_saturation_9p12_cosine_probe_ctxgold_per_sample.csv
```

跑前可确认注册表里是否有对应 `batch_id`：

```bash
python scripts/diagnostics/analyze_evidence_saturation.py \
  --registry-jsonl outputs/reports/run_registry.jsonl --list-batch-ids | head
```

当前 `run_eval/` 目录中的第二阶段核心入口：

- `run_end_to_end_batch.py`: 运行固定生成器的端到端 batch 评测  
- **烟测（mock、无 GPU）**：`configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k3.json` 与 `..._k4.json`，由 `tests/test_demo_ctxsel_k_smoke_batch.py` 覆盖，用于在 bump 默认 `context_select_k` 前确认 `k=3`/`k=4` 路径无 `generation_error`。
- `run_b_chain_phase2_three_arm.py`: **B 链 500** 三连跑（**rule** / **cosine_probe** / **oracle_item_leaves**），通过 `--generator-hf-model-name` 写入本机 Qwen 路径；支持 `--dry-run` 仅生成 `outputs/reports/tmp_phase2_configs/phase2_patch_*.json`。
- `compare_end_to_end_reports.py`: 汇总端到端 `EM / F1 / ROUGE-L` 与导航过程指标
- `export_end_to_end_diagnostics.py`: 从多个 batch 中导出对齐样本，便于人工诊断 `prompt / context / answer`
