# Mamba Tree-RAG / SSGS

本项目用于研究一种面向树状 RAG 的导航框架：使用 Mamba/SSM 作为 `Navigator`，通过状态快照与回溯实现低成本树搜索，并与 Transformer `Generator` 解耦。

当前阶段目标不是先追求大规模榜单结果，而是先搭建一个可运行、可审计、可复现的最小闭环系统。

## 当前研究主线

- 研究对象：树状 RAG 中的导航机制
- 核心机制：状态快照、回溯控制、导航-生成解耦
- 第一阶段目标：完整闭环框架与 trace 审计

详细研究说明见：

- `docs/research/SSGS_Research_Framework_CN.md`
- `docs/notes/Environment_Setup_CN.md`

## 推荐目录结构

```text
mamba2.1/
├─ README.md
├─ .gitignore
├─ docs/
│  ├─ research/
│  ├─ meetings/
│  ├─ papers/
│  └─ notes/
├─ configs/
├─ scripts/
├─ src/
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ cache/
├─ outputs/
│  ├─ runs/
│  ├─ reports/
│  └─ figures/
├─ notebooks/
└─ tests/
```

## 归档原则

- 根目录尽量只保留入口文件和全局说明
- 研究文档放到 `docs/`
- 代码放到 `src/`
- 运行脚本放到 `scripts/`
- 配置文件放到 `configs/`
- 数据放到 `data/`
- 实验结果与图表放到 `outputs/`

## Git 建议

建议尽早纳入 Git 仓库，但只提交轻量且可复现的内容：

- 提交：文档、代码、脚本、配置、测试
- 忽略：大数据、模型权重、缓存、运行日志、临时输出

这样后续无论本地开发、服务器迁移，还是上传 GitHub，都会更清晰。

## Phase A 快速运行

运行配置驱动的 Phase A 示例：

```powershell
py "scripts\run_nav\run_phase_a_pipeline.py"
```

运行批量导航示例：

```powershell
py "scripts\run_nav\run_navigation_batch.py"
```

比较导航汇总结果：

```powershell
py "scripts\run_nav\compare_navigation_reports.py"
```

运行最小演示脚本：

```powershell
py "scripts\run_nav\run_minimal_pipeline.py"
```

运行测试：

```powershell
py -m unittest discover -s tests -p "test_*.py"
```

默认示例会读取：

- `configs/experiment/phase_a_demo.json`
- `data/processed/demo_tree_payload.json`

批量示例会读取：

- `configs/experiment/navigation_batch_demo.json`
- `configs/experiment/navigation_batch_server_mamba_ssm_qwen.json`
- `configs/experiment/navigation_batch_server_mamba_ssm_qwen_cosine_probe.json`
- `data/processed/demo_navigation_batch.json`

当前默认导航器为：

- `navigator_type = mock`

这意味着系统现在已经具备正式的 `mamba2` 接入口，但默认仍用 mock 导航器先稳定跑通 pipeline。等到本地或服务器环境中的 `mamba_ssm` 依赖、模型加载和状态接口确认后，只需把配置切换为 `navigator_type = mamba2`，再补完真实前向逻辑即可进入真实 Mamba2 集成阶段。

运行结果会落盘到：

- `outputs/runs/<run_id>/run_payload.json`
- `outputs/runs/<run_id>/registry_row.json`
- `outputs/reports/run_registry.jsonl`
- `outputs/reports/navigation_summary.jsonl`
- `outputs/reports/batches/<batch_id>/batch_summary.json`

## 双环境建议

建议将导航器和生成器按环境拆分：

- 本地：`mamba2_native + qwen`
- 服务器：`mamba_ssm + qwen`

对应配置文件：

- `configs/experiment/local_native_qwen.json`
- `configs/experiment/server_mamba_ssm_qwen.json`

## 模型切换顺序

导航器的模型升级建议按下面顺序走：

1. 当前阶段：`mamba2-smoke`
2. 第一正式模型：`370M`
3. 第二正式模型：`1.4B`

建议不要直接跳到 `1.4B`。更稳的顺序是先让：

- 单样本运行稳定
- 批量导航稳定
- trace 和 routing 行为稳定

之后再切 `370M`；只有当 `370M` 的结果稳定、资源成本可控时，再切 `1.4B`。

当前推荐的正式模型入口：

- `configs\experiment\navigation_batch_server_mamba_370m_qwen_rule.json`
- `configs\model\mamba2_370m.example.json`
- `configs\model\mamba2_1p4b.example.json`
