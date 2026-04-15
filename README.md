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

按指定 `batch_id` 精确比较本轮结果：

```powershell
py "scripts\run_nav\compare_navigation_reports.py" --batch-id "your_batch_id"
```

导出 learned router 训练数据：

```powershell
py "scripts\run_nav\export_router_training_data.py"
```

训练 learned router：

```powershell
py "scripts\run_nav\train_learned_router.py"
```

把真实语料 `jsonl` 转成导航树：

```powershell
py "scripts\build_tree\build_tree_from_jsonl.py" --input "data\raw\your_corpus.jsonl" --output "data\processed\your_tree_payload.json"
```

把公开语料子集和 QA 标注一起转成 `tree payload + batch manifest`：

```powershell
py "scripts\build_tree\build_navigation_inputs_from_jsonl.py" --corpus-input "data\raw\your_corpus.jsonl" --qa-input "data\raw\your_qa.jsonl" --tree-output "data\processed\real_corpus_tree_payload.json" --batch-output "data\processed\real_corpus_navigation_batch.json"
```

把 wiki 风格长文档多跳样本先规整成 `corpus jsonl + qa jsonl`：

```powershell
py "scripts\build_tree\prepare_wiki_longdoc_subset.py" --input "data\raw\wiki_longdoc_samples.jsonl" --corpus-output "data\interim\wiki_longdoc_corpus.jsonl" --qa-output "data\interim\wiki_longdoc_qa.jsonl"
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

真实语料最小输入格式示例：

```json
{"doc_id":"doc_001","title":"Einstein Notes","summary":"Relativity overview","text":"Einstein proposed special relativity and general relativity."}
{"doc_id":"doc_002","title":"Newton Notes","summary":"Classical mechanics overview","text":"Newtonian mechanics explains force, motion, and gravity."}
```

与之对应的 QA `jsonl` 示例：

```json
{"sample_id":"q1","question":"What did Einstein propose?","reference_answer":"Einstein proposed special relativity and general relativity.","positive_doc_ids":["doc_001"]}
{"sample_id":"q2","question":"What does Newtonian mechanics explain?","reference_answer":"Newtonian mechanics explains force, motion, and gravity.","positive_doc_ids":["doc_002"]}
```

推荐做法：

- 主实验优先使用公开数据集或公开语料子集，不要用模型生成文本充当“真实语料”
- 手写小样本只用于 smoke test、接口验证和本地快速调试
- 先把公开数据规整成 `corpus jsonl + qa jsonl`，再统一走构建脚本，保证后续实验可复现
- 如果原始样本更接近多页 Wikipedia / 多节长文档格式，先运行 `prepare_wiki_longdoc_subset.py`，再运行 `build_navigation_inputs_from_jsonl.py`

`prepare_wiki_longdoc_subset.py` 输入样例：

```json
{"sample_id":"wiki_q1","question":"Where did the scientist work?","reference_answer":"At the Cavendish Laboratory.","supporting_page_ids":["page_scientist"],"pages":[{"page_id":"page_scientist","title":"Scientist","lead_text":"Scientist overview","sections":[{"section_id":"page_scientist__career","heading":"Career","paragraphs":["The scientist worked at the Cavendish Laboratory.","The scientist later taught at the university."]},{"section_id":"page_scientist__awards","heading":"Awards","paragraphs":["The scientist won a major prize."]}]}]}
```

这类样本会先被展开成：

- `corpus jsonl`：每个 section 一条记录
- `qa jsonl`：每个问题一条记录，正样本用 `supporting_section_ids` 或 `supporting_page_ids` 指定

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
- `configs\experiment\navigation_batch_server_mamba_370m_qwen_cosine_probe.json`
- `configs\experiment\navigation_batch_server_mamba_370m_qwen_learned_classifier.json`
- `configs\model\mamba2_370m.example.json`
- `configs\model\mamba2_1p4b.example.json`
