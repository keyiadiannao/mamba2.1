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
| MI-007 | 磁盘写满导致 `run_payload` / registry 写入失败（`OSError: 28`） | 已说明 |
| MI-008 | `never_visit_gold` 主导失配；root 层召回与候选池串联瓶颈 | 已处置（策略冻结） |

---

## MI-001：HF 镜像或网络故障导致整批生成失败

- **日期**：2026-04  
- **现象**：某批端到端结果中 `generation_error` 接近或等于样本总数（例如 `500/500`），EM/F1 等指标失去模型对比意义。日志或环境中可见 Transformers / 模型下载走 `hf-mirror` 等端点失败、超时或 TLS 错误。  
- **根因**：运行环境无法稳定访问模型权重或依赖的 Hub；与导航器、路由算法本身无关。  
- **涉及路径 / 配置**：生成器配置中的 Hub 端点、代理、离线缓存目录；`run_registry.jsonl` 中该 `batch_id` 对应条目的 `generation_error` 聚合。  
- **解决方案**：  
  1. 修复网络（代理、镜像、本机/服务器预下载并指向本地路径）。  
  2. 将该 `batch_id` 在实验结论中**单独标记为无效批次**，不参与与其它配置的优劣比较（见 MI-005）。  
  3. **已有本机模型快照时**：把配置里的 `generator_hf_model_name`（或代码支持的 `generator_model_path`）设为**含 `config.json` 与权重的本地目录**（例如 AutoDL 的 `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`），`transformers` 将直接 `from_pretrained(本地路径)`，不再访问 Hub / 镜像。亦可**不改 JSON**：对 `run_end_to_end_batch.py` 使用 **`--generator-hf-model-name <目录>`**，或导出环境变量 **`GENERATOR_HF_MODEL_NAME`**（CLI 优先于 env，二者均覆盖 JSON 中的同名字段）。若环境变量把端点指到不可达的 `hf-mirror.com`，应取消该设置或改为可用端点，否则仍可能在解析元数据时失败。  
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
  3. 若仅需同步少数文件，可用 **GitHub raw** 单文件覆盖（**不限于 `src/`**，`scripts/`、`tests/` 等同理）；URL 形如 `https://raw.githubusercontent.com/keyiadiannao/mamba2.1/main/<REL_PATH>`；网络不稳时在整段 URL 前加**你环境允许**的代理前缀（如 `https://ghproxy.net/https://raw.githubusercontent.com/...`，**域名以你侧可用与合规为准**）。  
  4. **推荐：容错拉取 + 定点覆盖 + 关键字校验**（与下面片段一致）：先 `git pull`（失败不退出），再对关键路径 `wget -O` 覆盖，最后用 `grep -n` 确认期望符号仍在。  
  5. 实操口径：若 `git pull` 因本地改动冲突中止，直接依赖 **3 + 4** 亦可把单文件对齐到远端 `main`；大目录仍用 **1 / 2** 与 ZIP，避免整仓 raw。  

**可复制片段示例**（路径、代理前缀、`main` 请按环境修改；在**仓库根**执行）：

```bash
cd ~/autodl-tmp/mamba2.1

git pull origin main || true

RAW_BASE='https://ghproxy.net/https://raw.githubusercontent.com/keyiadiannao/mamba2.1/main'

wget -O src/pipeline/phase_a_runner.py \
  "${RAW_BASE}/src/pipeline/phase_a_runner.py"
wget -O tests/test_phase_a_runner.py \
  "${RAW_BASE}/tests/test_phase_a_runner.py"

grep -n "question_overlap_topk" src/pipeline/phase_a_runner.py
grep -n "_select_context_items" src/pipeline/phase_a_runner.py
```

按需追加其它 `REL_PATH`（同一 `RAW_BASE` 规则），例如诊断脚本：`wget -O scripts/diagnostics/analyze_evidence_saturation.py "${RAW_BASE}/scripts/diagnostics/analyze_evidence_saturation.py"`。  

6. **能否把 ghproxy 当作「GitHub 镜像」以后只 `git pull`？**——**可以试，但不是 raw 那条链**：`wget` 用的是 **`raw.githubusercontent.com`**；`git pull` 走的是 **`github.com/.../.git`** 的 Git HTTP(S)。许多代理站支持把 **clone/fetch URL** 写成 **`https://<代理>/https://github.com/<org>/<repo>.git`**，从而 **`git pull` 走代理**。与单文件 `wget` **二选一或并存**即可：整仓能拉就用 **6**；拉不动或冲突仍用上面 **4** 的片段。  

**整仓 `origin` 走代理（示例，按需改域名与路径）**：

```bash
cd ~/autodl-tmp/mamba2.1

git remote -v
git remote set-url origin 'https://ghproxy.net/https://github.com/keyiadiannao/mamba2.1.git'
git fetch origin
git pull origin main
```

**若出现 `HTTP/2 stream ... was not closed cleanly: PROTOCOL_ERROR`**（经 `ghproxy` 等拉 `github.com` 时偶发）：在仓库目录执行 **`git config http.version HTTP/1.1`**（或 **`git config --global http.version HTTP/1.1`**）后重试 `fetch`/`pull`；仍失败则暂时 **`git remote set-url origin 'https://github.com/keyiadiannao/mamba2.1.git'`**（直连）或仅用 **4** 的 **`wget` raw** 补缺失文件。  

**注意**：第三方代理有**可用性 / 合规 / 单点故障**；部分环境 **`git push`** 经代理会失败，需改回 **`git@github.com:...`** 或直连 `https://github.com/...`；**LFS / submodule** 可能需额外配置。代理不可用时仍回到 **4** 的 `wget` 定点覆盖。**`git pull` 失败时**若跑批报错 **`FileNotFoundError` 缺 `configs/experiment/*.json`**，用 **4** 对 `REL_PATH` 逐条 `wget` 即可，不必等整仓拉通。  

7. **`git pull` 同时报「已跟踪文件本地修改将被覆盖」与「未跟踪文件将被覆盖」**：常见于先前 **`wget` 落在仓库内**、与远端**新纳入跟踪**的同名路径冲突，且 **`configs/*.json` 或文档**在服务器上被手改过。**以远端 `main` 为准**时：先把报错里列出的 **未跟踪** 路径移出仓库（如 **`mkdir -p /tmp/mamba_git_bak && mv <路径…> /tmp/mamba_git_bak/`**），再 **`git restore --worktree -- <已跟踪路径或目录>`** 丢弃本地修改，最后 **`git pull origin main`**。**须保留本地改动**时：先 **`git stash push -u -m autodl`**，再 **`git pull`**，最后 **`git stash pop`**（可能有冲突需手解）。  

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
  - `question_entity_match_topk`：按 `extract_question_entities` + `compute_entity_match_score`（命中问题实体占提取实体数的比例）降序；**次级键**为与 `question_overlap_topk` 相同的词重叠数，再按原序 tie-break。无实体时退化为仅靠词重叠（与 overlap 行为一致）。  
  - **`context_select_pool_max_items`（可选）**：仅当 **`context_select_mode` 非 `off`** 时生效；为 **正整数** 时，构建 context 的条数上限取 **`max(context_max_items, context_select_pool_max_items)`**，再经 `context_select_*` 截断到 **`context_select_k`**；用于在**不**改 Controller 的前提下扩大「visited→打分」候选池（**B2**）。为 `off` 或未设置时行为与改前一致。  
- **仓库模版默认（2026-04-16，2026-04-18）**：`configs/experiment/` 下 **`question_overlap_topk`** 与 `t1_visited_leaves_ordered` / `flat_leaf_concat` 组合时 **`context_select_k=4`**（`first_k3`/`dedupe_k3` 例题仍为 `3`）；`oracle_item_leaves` 例题显式 `context_select_mode=off`。  
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

## MI-007：磁盘空间耗尽导致端到端批中途失败

- **日期**：2026-04  
- **现象**：运行 `run_end_to_end_batch.py` 或串联脚本时，在写入 `outputs/runs/<run_id>/run_payload.json` 或 `run_registry.jsonl` 过程中抛出 **`OSError: [Errno 28] No space left on device`**，子进程以非零退出；**已完成的样本**可能部分落盘，**当前 `batch_id` 往往不完整**，不应直接当作主表结论。  
- **根因**：系统盘或挂载盘（含 **`/root`**、**`/root/autodl-tmp`** 等）**可用块耗尽**；每条 run 含 payload、trace、prompt，**500×Qwen7B** 批会快速占满磁盘。  
- **涉及路径 / 配置**：`outputs/runs/`、`outputs/reports/`、`outputs/reports/tmp_phase2_configs/`；可选将 **`output_dir` / `batch_output_dir`** 改到更大数据盘（若平台支持）。  
- **解决方案**：  
  1. **`df -h`** 确认满的是哪一块挂载；**`du -sh outputs/runs outputs/reports`** 看占用。  
  2. **归档或删除**不再需要的旧 `outputs/runs/*`、旧 `end_to_end_batches/*`、HF 缓存重复副本、容器内无关大包。  
  3. 大实验前预留 **≥ 数十 GB**（与模型体积、批大小、是否保留全量 `run_payload` 成正比）；必要时 **只保留 `batch_summary.json` + 抽样 `run_payload`** 的运维策略。  
  4. 串联多臂时：**每臂跑前检查剩余空间**；上一臂跑完后若空间仍紧，先清理再跑下一臂。仓库脚本 **`scripts/run_eval/run_b_chain_phase2_three_arm.py`** 在剩余 **< 5 GiB** 时会在 stderr 打出警告（仍会继续，需人工判断是否中止）。  
- **验证**：`df -h` 显示目标挂载有足够余量后，从失败臂 **重新跑**（或从断点样本继续，需自行脚本化）；新批 `generation_error` 与 `sample_count` 正常闭合。

---

## MI-008：`never_visit_gold` 主导失配与「看见但进不了 context」串联瓶颈

- **日期**：2026-04  
- **现象**（`pilot200` 导航批）：
  - 基线（`cos0.7, probe0, e8, pool8`）中 `never_visit_gold` 约 `61%`。
  - 进一步分解 `never_visit_gold`：`root-miss`（金叶所在 root 子枝完全未访问）占比约 **98.4%**，`in-root-miss` 约 `1.6%`。
  - 开启更激进 root 探测（`probe2`）可降低 `never_visit_gold`，但新增样本主要转为 `visit_gold_but_outside_pool`，`in_context` 不升或回退。
  - 仅扩大 `pool` 而不稳定提升 root 召回，收益有限；仅提升 root 召回而不放宽候选漏斗，收益也被截断。
- **根因**：瓶颈是**串联**的两段：
  1. **根层召回不足**（主矛盾，`root-miss` 主导）；  
  2. **候选池 / context 漏斗截断**（次矛盾，新增召回未转化为 `in_context`）。
- **涉及路径 / 配置**：
  - `src/controller/ssgs_controller.py`（`explore_root_probe_top_m`、`explore_root_probe_budget_per_child`、`explore_top_m_root_children`）
  - `src/pipeline/phase_a_runner.py`（`context_select_pool_max_items` 接线）
  - 导航批配置：`router_cosine_weight`、`max_evidence`、`context_select_pool_max_items`、probe 开关。
- **解决方案（冻结口径）**：
  1. 先冻结启发式最优基线：`cos0.7 + probe1 + e8 + pool20`（用于后续 A/B 与回归基线）。  
  2. 停止继续扩启发式大网格；把主线转入**学习式 root 路由**（目标直接打 `root-miss`）。  
  3. 评估门槛固定为：`never_visit_gold` 下降 + `in_context` 上升，且 `B3->B1 <= B1->B3`。
- **补充（2026-04，已验证）**：在真实语料 **全 fan-out** 下，**仅线性 root 头**（`learned_root_classifier` 且 **`learned_root_blend_alpha=1`**）会出现金叶访问近零；**必须与 `RuleRouter` 同特征下的规则分数混合**：配置键 **`learned_root_blend_alpha`**。**`500` 上 `α=0.5` 与 `0.25` 金叶指标同量级、`nav_ms` 显著更优**，工程默认 **`0.5`**（约 50% rule + 50% learned）；显式 **`0.25`** 仍可复现旧默认。训练数据仍用 **`--root-only` + `--max-root-children 128`** 导出与 **listwise** checkpoint；**扫 `α` 或做 rule 对照时勿换训练 jsonl/checkpoint**，除非单独开「训练数据消融」。详见 **`docs/research/Navigation_Experiment_Record_CN.md` §6.5**。
- **验证**：
  - 导航批固定 `pilot200` 对比表：`never_visit_gold / outside_pool / in_pool_not_context / in_context` + `nav_ms`。  
  - 若学习式 root 路由在相同预算下稳定降低 `root-miss`，再进入端到端批验证 EM/F1 传导。

---

## 修订历史

| 日期 | 说明 |
|------|------|
| 2026-04-16 | 初版：从研究文档迁出重大问题叙述，建立专档与交叉引用约定。 |
| 2026-04-16 | 追加 MI-006 的 500 样本 A/B 验证结果（`off` vs `question_overlap_topk(k=3)`，两组 `generation_error=0/500`）。 |
| 2026-04-16 | `configs/experiment/` 主流模版写入 `context_select_mode` 默认值；Oracle 臂为 `off`。 |
| 2026-04-16 | MI-002 补充：记录 `git pull` 被本地改动阻塞时的容错同步口径与典型冲突文件。 |
| 2026-04-16 | MI-001 补充：`generator_hf_model_name` 指向本机模型目录以绕过失效 `hf-mirror`。 |
| 2026-04-18 | MI-006：仓库 overlap 默认 `context_select_k` bump 至 `4`；增加 demo 烟测配置与 `tests/test_demo_ctxsel_k_smoke_batch.py`。 |
| 2026-04-18 | 新增 MI-007：磁盘满 `Errno 28` 与端到端批处置（正文已含串联脚本低余量 stderr 警告）。 |
| 2026-04-17 | 新增 MI-008：确认 `root-miss` 为 `never_visit_gold` 主因；冻结启发式终版并切换学习式 root 路由主线。 |
| 2026-04-17 | MI-008 补充：`learned_root_blend_alpha` 与 rule 分数混合为当前可用学习式 root 形态；记录见 `Navigation_Experiment_Record_CN.md` §6.5。 |
| 2026-04-17 | MI-008：`500` 上 `α=0.5` 相对 `0.25` 金叶同量级、`nav_ms` 更优，工程默认 `learned_root_blend_alpha` 改为 **`0.5`**。 |
| 2026-04-17 | MI-008：`α>0.5` 不设为常规必扫项（`α→1` 高风险、边际收益低）；例外见 `Navigation_Experiment_Record_CN.md` §6.5 末段。 |
| 2026-04-17 | 端到端 EM 全零：manifest 中 **`reference_answer` 为 list** 时旧逻辑未规范化；已用 **`normalize_reference_for_scoring`** 修复（见 `Navigation_Experiment_Record_CN.md` §6.6 排错）。 |
| 2026-04-17 | P0 端到端 `500`：`learned_root blend0.5` 相对 `rule` frozen，**EM 0.200 vs 0.186**，见 `Navigation_Experiment_Record_CN.md` §6.6 表。 |
| 2026-04-18 | P0-2 导航批 `N=200`：金叶 **visited 0.445 vs 0.41**、`gold_missing` **122 vs 130**；检索 EM **0.125 vs 0.11**；`batch_id` 见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | P1 读侧第一步：导航批模版 **`navigation_batch_real_corpus_p1_rule_frozen_nav_reg200_pool32.example.json`**（`pool=32` 对照 P0-2 `pool=20`）；执行顺序与 ctx-gold 命令见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | P1-1 实测 **`nav_p1_reg200_rule_pool32_20260418_022308Z`**：`pool` 32 与 P0-2 rule **`pool=20` 金叶/检索 EM 同值**，见 `Navigation_Experiment_Record_CN.md` §6.6；下一读侧臂改试 **`k` / `mode`**。 |
| 2026-04-18 | P1-2 导航批模版 **`navigation_batch_real_corpus_p1_rule_frozen_nav_reg200_overlap_k5.example.json`**（**`k=5`** 单臂对照 **`k=4`**）；命令见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | P1-2 实测 **`nav_p1_reg200_rule_overlap_k5_20260418_023920Z`**：**`k=5` 与 `k=4` 金叶/检索 EM 同值**；下一读侧臂 **`context_select_mode`**，见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | P1-3 导航批模版 **`navigation_batch_real_corpus_p1_rule_frozen_nav_reg200_entity_match_k4.example.json`**（**`question_entity_match_topk`**）；命令见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | 导航批 **Oracle 上界** 模版 **`navigation_batch_real_corpus_nav_reg200_oracle_item_leaves.example.json`**（`context_source=oracle_item_leaves`，诊断用）；见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | P1-3 实测 **`nav_p1_reg200_rule_entity_match_k4_20260418_030137Z`**：金叶与 overlap **同值**；检索 EM **0.12 vs 0.11**（`N=200`），见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | 主线回到**导航侧由前到后**（非 root 路由 → Controller 探索预算 → 接受侧 P0-A′）；**已有 Oracle 500 e2e 则默认不必**再跑导航批 Oracle 200；见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **Accept 门审计**：`scripts/diagnostics/audit_accept_gate.py` + `src/diagnostics/accept_gate_audit.py`（按 `run_registry.jsonl` + `batch_id` 读 `run_payload.json`）；用法见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | P0 端到端 500：`audit_accept_gate` 显示 **~55%～57% 样本从未 visit 金叶**；visit 未 accept 叶次上 **`reject_leaf_branch_cap` > `min_relevance`**；表见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **P0-A′**：已加端到端模版 **`…p0_rule_frozen_nav_probe_budget2.example.json`** / **`…p0_learned_root_blend05_probe_budget2.example.json`**（仅 **`explore_root_probe_budget_per_child: 2`**）；执行与判读见 `Navigation_Experiment_Record_CN.md` §6.6 **P0-A′** 段。 |
| 2026-04-18 | **P0-A′**：补充仅导航批模版 **`navigation_batch_real_corpus_p0_probe_budget2_{rule,learned_root_blend05}.example.json`**（不跑 7B）；§6.6 增加 AutoDL 可复制命令（避免 `cd … \|\| exit 1` 导致 shell 直接退出）。 |
| 2026-04-18 | **P0-A′** 导航 **`n=10`** 烟测台账（`nav_p0_probe_budget2_rule_20260418_040256Z` / `…learned_root_blend05_20260418_040319Z`）与读法见 `Navigation_Experiment_Record_CN.md` §6.6；**不得**与 500 主结论混用。 |
| 2026-04-18 | **P0-A′**：§6.6 补 **导航 `n=200` / 满 manifest** 可复制命令块（含 saturation + `audit_accept_gate` 循环）；烟测表补全 rule 的 `audit` 与 learned 的 saturation。 |
| 2026-04-18 | **P0-A′** 导航 **满 manifest（`sample_count=500`）**、`probe_budget=2`：`reject_leaf_branch_cap` 叶次 **≈44/41**（对 e2e 基线 **≈85/76**）；**`never_visit_any_gold`** 仍 **≈0.58/0.55**；表与判读见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **P0-A′** `rule` 导航满 500 **复跑** `nav_p0_probe_budget2_rule_20260418_045515Z`：摘要与 `…041200Z` 一致；见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **P0-A′ 严对照**：新增导航模版 **`navigation_batch_real_corpus_p0_probe_budget1_{rule,learned_root_blend05}.example.json`**（**`explore_root_probe_budget_per_child=1`、满 manifest**）；与 **`probe2`** 复跑区分；命令见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **P0-A′** 导航 **`probe1` 满 500** 已跑：`nav_p0_probe_budget1_rule_20260418_051729Z`、`…learned_root_blend05_20260418_053127Z`；**`audit` 与 P0 e2e 500 逐字段一致**；**`probe1`→`probe2`** 并排见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **P0-A′ 执行项**：端到端 **`probe_budget2` 满 500** 两臂 + 诊断命令与结果表占位见 `Navigation_Experiment_Record_CN.md` §6.6；判停 **MI-004/005**。 |
| 2026-04-18 | **P0-A′ e2e `probe2` 500** 已跑：`…rule_probe_budget2_20260418_060702Z`、`…learned_root_blend05_probe_budget2_20260418_062859Z`；过程/`audit` 与导航 `probe2` 同向；**EM/F1** 见 `batch_summary.json` 或 §6.6 **`jq`**。 |
| 2026-04-18 | **P0-A′** 文档勘误：**`probe_budget2` 台账为导航满 manifest（500）**；严对照缺 **`probe_budget=1` 导航满量**两条，`n=200` 的 P0-2 **不可替代**；见 `Navigation_Experiment_Record_CN.md` §6.6。 |
| 2026-04-18 | **`run_navigation_batch` / `run_end_to_end_batch`** 增加可解析行 **`__SSGS_BATCH_ID__=`**（便于终端 `sed` 取 `batch_id`）；**不**为烟测维护额外 shell，用法见 `Navigation_Experiment_Record_CN.md` §6.6。 |
