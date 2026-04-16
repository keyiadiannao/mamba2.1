# `scripts/` 脚本目录约定

- `build_tree/`: 构建树索引、摘要树、预处理脚本
- `run_nav/`: 导航阶段运行脚本
- `run_eval/`: 评测、汇总、导出报告脚本
- `utils/`: 数据搬运、检查、转换等辅助脚本

建议把“可重复执行的流程”写成脚本放这里，不要把一次性命令长期留在终端历史里。

当前 `run_eval/` 目录中的第二阶段核心入口：

- `run_end_to_end_batch.py`: 运行固定生成器的端到端 batch 评测  
- **烟测（mock、无 GPU）**：`configs/experiment/end_to_end_batch_demo_smoke_ctxsel_overlap_k3.json` 与 `..._k4.json`，由 `tests/test_demo_ctxsel_k_smoke_batch.py` 覆盖，用于在 bump 默认 `context_select_k` 前确认 `k=3`/`k=4` 路径无 `generation_error`。
- `run_b_chain_phase2_three_arm.py`: **B 链 500** 三连跑（**rule** / **cosine_probe** / **oracle_item_leaves**），通过 `--generator-hf-model-name` 写入本机 Qwen 路径；支持 `--dry-run` 仅生成 `outputs/reports/tmp_phase2_configs/phase2_patch_*.json`。
- `compare_end_to_end_reports.py`: 汇总端到端 `EM / F1 / ROUGE-L` 与导航过程指标
- `export_end_to_end_diagnostics.py`: 从多个 batch 中导出对齐样本，便于人工诊断 `prompt / context / answer`
