# 重大问题与解决方案（专档）

本文档为**工程与实验归因类重大问题**的**唯一详尽记录处**：现象、根因、处置与验证方式均写在此处；其它文档（含 `docs/research/` 下研究框架与实验记录）仅保留**一句话索引**或**指标表**，不在别处重复展开同类叙述。

更新规范：每条目含 **编号**、**日期**、**现象**、**根因**、**涉及路径 / 配置**、**解决方案**、**验证**；如有关联提交，在文末注明 `commit` 或本机 `git rev-parse HEAD` 与 ZIP 来源。

---

## 索引

| 编号 | 主题 | 状态 |
|------|------|------|
| MI-001 | Hugging Face 镜像 / 网络不可达导致整批 `generation_error` | 已处置 |
| MI-002 | 服务器侧无 `.git` 或与 GitHub 不同步时的代码更新 | 已处置 |
| MI-003 | 诊断脚本对 `tree_path`、`positive_leaf_indices` 与 trace 字段的依赖 | 已说明 |
| MI-004 | 导航 / 证据 recall 提升与生成 readout 的 trade-off；盲扫阈值 | 已处置 |
| MI-005 | 「无效批次」不得进入配置优劣比较 | 已说明 |
| MI-006 | 证据上下文排序与截断：`context_select_mode` | 已实现 |

---

## MI-001：HF 镜像或网络故障导致整批生成失败

- **日期**：2026-04  
- **现象**：某批端到端结果中 `generation_error` 接近或等于样本总数（例如 `500/500`），EM/F1 等指标失去模型对比意义。日志或环境中可见 Transformers / 模型下载走 `hf-mirror` 等端点失败、超时或 TLS 错误。  
- **根因**：运行环境无法稳定访问模型权重或依赖的 Hub；与导航器、路由算法本身无关。  
- **涉及路径 / 配置**：生成器配置中的 Hub 端点、代理、离线缓存目录；`run_registry.jsonl` 中该 `batch_id` 对应条目的 `generation_error` 聚合。  
- **解决方案**：  
  1. 修复网络（代理、镜像、本机/服务器预下载并指向本地路径）。  
  2. 将该 `batch_id` 在实验结论中**单独标记为无效批次**，不参与与其它配置的优劣比较（见 MI-005）。  
  3. **已有本机模型快照时**：把配置里的 `generator_hf_model_name`（或代码支持的 `generator_model_path`）设为**含 `config.json` 与权重的本地目录**（例如 AutoDL 的 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`），`transformers` 将直接 `from_pretrained(本地路径)`，不再访问 Hub / 镜像。若环境变量把端点指到不可达的 `hf-mirror.com`，应取消该设置或改为可用端点，否则仍可能在解析元数据时失败。  
- **验证**：同配置在修复网络后复跑小批（如 20 条）应使 `generation_error` 归零或降至可接受比例；`run_payload.json` 中可抽查单条 `generation_error` 字段。

---

## MI-002：服务器上 Git 不可用或与 GitHub 同步困难

- **日期**：2026-04  
- **现象**：容器内 `git clone` / `git pull` 频繁超时；或部署方式为 **GitHub `main` 分支 ZIP 解压**，目录内**无 `.git`**，无法在服务器上直接 `git rev-parse HEAD`。  
  补充：即便可连通，`git pull` 也可能因服务器本地改动而中止（`Your local changes to the following files would be overwritten by merge`），常见于 `scripts/diagnostics/analyze_evidence_saturation.py`、`src/pipeline/phase_a_runner.py`、`tests/test_phase_a_runner.py` 等被临时改过的文件。  
- **根因**：网络策略、镜像站不稳定或运维选择「上传包」而非 clone。  
- **涉及路径 / 配置**：整仓解压目录；需与代码分离拷贝的 `data/`、`outputs/`。  
- **解决方案**：  
  1. 本机打包前执行 `git rev-parse HEAD`，在实验记录或 `run_registry` 旁注记**提交哈希**。  
  2. 大目录从旧工作区拷入新解压目录，避免重复下载数据。  
  3. 若仅需同步少数文件，可用稳定代理拉取单文件 raw（例如 `ghproxy` 等，以你环境合规性为准）覆盖 `src/` 下对应路径。  
  4. 实操口径：先尝试常规拉取；若因本地改动冲突导致拉取中止，则切换到“**容错拉取 + 关键文件定点覆盖 + 关键字校验**”流程，保证最小改动恢复到目标提交行为。  
- **验证**：服务器上关键文件与提交内容一致（哈希或 diff）；实验复跑与本地同提交结果可对照。

---

## MI-003：证据饱和 / 金叶子诊断脚本的数据前提

- **日期**：2026-04  
- **现象**：运行 `scripts/diagnostics/analyze_evidence_saturation.py` 时部分指标为空、不可比，或与人工对 trace 的预期不一致。  
- **根因**：脚本依赖从 `run_registry.jsonl` 定位的 `run_payload.json` 中读取树与 trace；**批次需传入 `positive_leaf_indices`（映射到 trace 的 `leaf_indices_required`）** 才能计算金叶子访问 / 接受类指标；`tree_path` 必须在 registry 与 payload 链路中可解析。  
- **涉及路径 / 配置**：`scripts/diagnostics/analyze_evidence_saturation.py`；批量入口传入的 `tree_path`、`positive_leaf_indices`。  
- **解决方案**：生成批量配置时保证每条样本的 `tree_path` 与树 JSON 一致；需要金指标时传入 `positive_leaf_indices`。可选 `--with-context-gold-metrics` 区分「导航证据」与「生成器实际 context」。  
- **验证**：对已知 `batch_id` 跑脚本，摘要 JSON 中非空字段与 CSV 抽样与单条 `run_payload.json` 一致。

---

## MI-004：证据 recall 与生成 readout 的 trade-off（含「扩上下文反而变差」）

- **日期**：2026-04  
- **现象**（同一主表约 500 条上的诊断摘要）：  
  - **Oracle** 与 **Rule** 的生成器上下文含金文比例差距极大：`oracle` 的 `mean_frac_gold_leaf_texts_in_generator_context` 约 `0.999`、`all_gold` 约 `0.994`；`rule` 仅在约 `0.128~0.158`，主瓶颈仍在「生成器看到的证据质量」而非「导航是否跑完」。  
  - 调高导航相关参数（如扩大 `n_context` / 证据预算）时，可能出现**过程指标**（访问 / 接受 gold、ctx-gold）略升，但 **500 条端到端 EM/F1 下降**（例如某组 `pem3 + overlap_off + mrs2.0` 相对基线下降至约 `0.148 / 0.168`）。  
  - 样本级分桶示例：`A(ctx↑,F1↑)=0.6%`、`B(ctx↑,F1↓)=0.8%`、`C(ctx≈,F1↓)=7.4%`、`D=91.2%`，正向收益不足。  
  - 在大量 worst 样本中，观察到 `n_context` 从 `3~5` 扩到 `8` 后出现 **F1 从 `1.0` 变为 `0.0`**，更符合**上下文噪声 / 顺序效应**而非「阈值未扫够」。  
- **根因**：在固定 readout 与 prompt 下，单纯堆证据会引入排序与噪声；与「导航未完成」相比，**evidence-to-generation 消费链路**更紧迫。  
- **涉及路径 / 配置**：`src/pipeline/phase_a_runner.py`（`context_max_items`、`max_evidence`、Controller 的 `min_relevance_score` / `max_depth` 等）；生成器与后处理配置。  
- **解决方案**：  
  1. **冻结**以 `mrs` / `pem` 为主轴的盲目细扫；将单变量主线转到**证据排序与截断**（见 MI-006）。  
  2. 每轮同时看**终点指标**（EM/F1/ROUGE-L）与**过程指标**（ctx-gold、gold visited/accepted、分桶），若出现「过程升、终点降」，按 MI-005 判停，不写入主表结论。  
  3. 用 `scripts/diagnostics/analyze_evidence_saturation.py` 看 `frac_evidence_budget_saturated`、`mean_unique_entities_in_evidence`、金叶子访问与接受比例，再决定 anti-collapse、接受阈值或探索顺序的下一刀。  
- **验证**：`context_select_mode` 消融（如 `off` vs `question_overlap_topk`）在固定 500 条上报告终点 + 过程指标；烟测批需标明为小样本提示，不与主表混写。

---

## MI-005：无效批次不得进入配置比较

- **日期**：2026-04  
- **现象**：某批因网络、依赖或整批跳过生成等原因，指标与正常批不可比。  
- **根因**：失败归因在「环境 / 运维」而非算法配置。  
- **解决方案**：在实验记录与汇总脚本中显式标记该 `batch_id` 为**无效批次**；撰写结论时仅对比「`generation_error` 可接受」的批次。  
- **验证**：汇总前扫描 `run_registry.jsonl` 或 per-run `generation_error` 计数。

---

## MI-006：上下文后处理 `context_select_mode`（不改 Controller 主逻辑）

- **日期**：2026-04  
- **现象**：需在**不修改**导航 Controller 核心的前提下，验证「排序优先于单纯访问顺序」对 readout 的影响。  
- **根因**：MI-004；需要可审计、可回滚的最小侵入开关。  
- **涉及路径 / 配置**：`src/pipeline/phase_a_runner.py` 中 `_select_context_items`；配置键 `context_select_mode`、`context_select_k`。  
- **解决方案**：支持模式：  
  - `off`：**代码未传键时**的默认（不后处理）。  
  - `first_k`：按当前顺序截断前 `k` 条。  
  - `dedupe_entity_then_k`：按实体键去重后取前 `k` 条。  
  - `question_overlap_topk`：按问题词与证据文本词重叠数降序，稳定 tie-break 后取 top `k`。  
- **仓库模版默认（2026-04-16）**：`configs/experiment/` 下含 `t1_visited_leaves_ordered` / `flat_leaf_concat` 的 JSON 已写入 `question_overlap_topk` + `context_select_k=3`；`oracle_item_leaves` 例题显式 `context_select_mode=off`，避免动 Oracle 顺序。  
- **验证**：单元测试 `tests/test_phase_a_runner.py`；端到端小批 `generation_error` 为零的对照与 A/B 批。

### MI-006 验证更新（2026-04-16，500 样本主表）

在同一规模（`sample_count=500`）且 `generation_error=0/500` 的可比前提下，`question_overlap_topk (k=3)` 相对 `off` 有稳定提升：

| 实验臂 | batch_id | EM | F1 / ROUGE-L | nav_success |
|---|---|---:|---:|---:|
| `off` | `ab500_ctxsel_off_localqwen_v2_20260416_101040Z` | `0.170` | `0.1916` | `1.0` |
| `question_overlap_topk(k=3)` | `ab500_ctxsel_overlap3_localqwen_20260416_103201Z` | `0.194` | `0.2116` | `1.0` |

- **增益（overlap3 - off）**：`EM +0.024`（约 +14.1% 相对提升），`F1 +0.0201`（约 +10.5% 相对提升）。
- **结论**：在不改 Controller 主逻辑条件下，`context_select_mode=question_overlap_topk` 在 500 主表上验证通过，可作为后续 readout-first 主线的默认候选。

---

## 修订历史

| 日期 | 说明 |
|------|------|
| 2026-04-16 | 初版：从研究文档迁出重大问题叙述，建立专档与交叉引用约定。 |
| 2026-04-16 | 追加 MI-006 的 500 样本 A/B 验证结果（`off` vs `question_overlap_topk(k=3)`，两组 `generation_error=0/500`）。 |
| 2026-04-16 | `configs/experiment/` 主流模版写入 `context_select_mode` 默认值；Oracle 臂为 `off`。 |
| 2026-04-16 | MI-002 补充：记录 `git pull` 被本地改动阻塞时的容错同步口径与典型冲突文件。 |
| 2026-04-16 | MI-001 补充：`generator_hf_model_name` 指向本机模型目录以绕过失效 `hf-mirror`。 |
