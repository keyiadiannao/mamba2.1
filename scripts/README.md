# `scripts/` 脚本目录约定

- `build_tree/`: 构建树索引、摘要树、预处理脚本
- `run_nav/`: 导航阶段运行脚本
- `run_eval/`: 评测、汇总、导出报告脚本
- `utils/`: 数据搬运、检查、转换等辅助脚本

建议把“可重复执行的流程”写成脚本放这里，不要把一次性命令长期留在终端历史里。

`diagnostics/`（过程指标，**不启动评测**）：

- **`analyze_evidence_saturation.py`**：按 `run_registry.jsonl` 的 **`batch_id`**（或 `glob`）聚合 `run_payload.json`；加 **`--with-context-gold-metrics`** 时加载每条的 **`tree_path`**。用法见脚本顶部 docstring 与专档 **MI-003**；**非关键、低复用的一次性命令**不写入本 README，避免与实验记录双处维护。

当前 `run_eval/` 目录中的第二阶段核心入口：

- `run_end_to_end_batch.py`: 运行固定生成器的端到端 batch 评测  
- **烟测（mock、无 GPU）**：`configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k3.json`、**`..._overlap_k4.json`**、**`..._entity_k4.json`**（`question_entity_match_topk`），由 `tests/test_demo_ctxsel_k_smoke_batch.py` 覆盖。  
- **B1 全量 500（与 §9.12 `rule` 仅差 `context_select_mode`）**：`configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_rule_ctxsel_entity_match_k4.example.json`（`question_entity_match_topk` + `k=4`）；本机 Qwen 路径仍用 **`run_end_to_end_batch.py --config ...`** 注入或编辑 JSON 内 **`generator_hf_model_name`**。  
- **B2（扩大打分池）**：在同上协议上增加 **`context_select_pool_max_items`**（例 **`…entity_match_k4_pool20.example.json`**，`pool_max=20`）。
- `run_b_chain_phase2_three_arm.py`: **B 链 500** 三连跑（**rule** / **cosine_probe** / **oracle_item_leaves**），通过 `--generator-hf-model-name` 写入本机 Qwen 路径；支持 `--dry-run` 仅生成 `outputs/reports/tmp_phase2_configs/phase2_patch_*.json`。
- `compare_end_to_end_reports.py`: 汇总端到端 `EM / F1 / ROUGE-L` 与导航过程指标
- `export_end_to_end_diagnostics.py`: 从多个 batch 中导出对齐样本，便于人工诊断 `prompt / context / answer`
