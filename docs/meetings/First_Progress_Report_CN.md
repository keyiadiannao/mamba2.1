# 阶段进展汇报（第一次）

**课题方向**：树状 RAG 导航（Mamba 状态 + 回溯控制）与生成器解耦的**可运行、可审计、可复现**闭环。  
**汇报目的**：对齐问题定义、展示已交付工程与关键数据、说明瓶颈与**下一刀**（不夸大终点收益）。  
**建议时长**：口头 **12～18 分钟**（可按听众删减「附录」与部分表格）。

---

## 1. 一句话与边界

- **一句话**：在真实长文档树上，用固定规模导航器与显式控制逻辑完成**多步探索与回溯**，输出可追溯证据链，再由**固定生成器**完成读证据答题；本阶段优先证明**系统与协议成立**，再讨论导航改动向 **EM/F1** 的稳定传导。  
- **明确不主张**：尚未充分证明「导航单项改动即可带来大规模榜单提升」；不把 **Oracle 作弊上界** 与真实导航臂混谈。  
- **叙事原则**：**过程指标与终点指标并列**；过程升、终点降时**不写入主结论**（与仓库 **MI-004 / MI-005** 一致）。

---

## 2. 系统结构（汇报用「四块」）

| 模块 | 职责 | 备注 |
|:---|:---|:---|
| 数据与 manifest | 真实子集、树 payload、样本级 `positive_leaf_indices` | 批量实验统一 manifest，便于对齐 |
| Navigator + Router + Controller | 读节点、打分排序、DFS/回溯、预算与接受门 | 与 **Generator** 解耦 |
| 落盘与 registry | `run_registry.jsonl`、`run_payload`、批次 summary | 支持复现与事后审计 |
| 诊断与审计 | 证据饱和、**`audit_accept_gate`**、分桶摘要等 | 区分 **从未 visit 金叶** vs **visit 后未进 accept** |

---

## 3. 阶段内已交付（可验收）

1. **端到端链路**：真实 manifest 上可跑导航批与（按需）端到端批；关键配置在 `configs/experiment/`。  
2. **过程审计**：`scripts/diagnostics/audit_accept_gate.py` 等，可对 **`never_visit` / `visit_miss`（visit 金叶但未满足 accept）** 等做样本级统计。  
3. **导航侧主进展（满 500 主表口径）**：在 **`rule` + visit-rule + 实体偏置** 扫参后，将默认工作点收敛至 **`122155Z`（`entity_boost_alpha=0.3`，`max_nodes=80`，`probe_budget=2`）**；相对早期 **`probe2` 纯 `rule` 满 500（`041200Z`）**，**`never_visit`（从未 visit 任一金叶）由约 0.58 降至约 0.38**，且 **`visit_miss` 仍满足 ≤0.12 的过程门**。  
4. **单变量纪律**：同一旋钮上 **`max_nodes` / `max_depth` / `probe_top_m`** 等烟测结论已收口（硬顶非主因、`probe_top_m=2` 过门等），详见 **`docs/research/Navigation_Experiment_Record_CN.md` §6.7**。  
5. **文档与专档**：实验事实以 **`Navigation_Experiment_Record_CN.md`** 为准；判停与工程归因以 **`docs/Major_Issues_And_Resolutions_CN.md`** 为准；研究框架以 **`docs/research/SSGS_Research_Framework_CN.md`** 为准。

---

## 4. 关键数字表（汇报主表）

### 4.1 导航主锚（满 manifest，当前 500）

| 对比项 | 代表 `batch_id`（后缀） | `never_visit`（audit） | `visit_miss`（约） | 备注 |
|:---|:---|---:|---:|:---|
| **`probe2` 纯 `rule`（实体偏置前）** | `…041200Z` | **~0.58** | ~0.12 | 旧主矛盾锚点 |
| **实体偏置默认工作点** | `…122155Z` | **~0.38** | **~0.11** | **当前 `rule` 侧默认候选**（保守对照 `105756Z`） |

*口径说明：`never_visit` / `visit_miss` 以 **`accept_gate_audit_*.json`** 顶层字段为准；与早期 **`pilot200` / `never_visit_gold`** 等字段**勿混读**（不同协议与命名）。*

### 4.2 「④ 逼近 Oracle」单变量烟测（`n=200`，与 `122155Z` 同栈，2026-04-19）

**目的**：在**非作弊**（不设 `oracle_item_leaves`、不注入 `leaf_indices_required`）前提下，观察「真实导航」下单旋钮对 **金叶可达性** 与 **accept 门** 的影响；**硬门**：**`visit_miss` ≤ 0.12**。

| 单变量臂 | `batch_id` | `never_visit` | `visit_miss` | 叶级 disposition（摘要） | 结论（汇报用语） |
|:---|:---|---:|---:|:---|:---|
| **`max_evidence`: 12→14** | `nav_p0_visit_rule_entity_boost_a030_abl_maxev_14_20260419_042514Z` | **0.39** | **0.10** | `branch_cap` 25 / `minrel` 4 | 与 **`122155Z` 烟测档 ~0.40 / ~0.10** 同量级，**过过程门** → **建议满 500 复核** |
| **`routing_mode`: `cosine_probe`** | `nav_p0_visit_rule_entity_boost_a030_cosine_probe_20260419_044414Z` | **0.20** | **0.19** | `branch_cap` 41 / `minrel` 9 | **`never_visit` 大赢**，但 **`visit_miss` 顶穿 0.12** → **不替换默认**；属 **trade-off** 备忘 |
| **`learned_root` + `α=0.5`** | `nav_p0_visit_rule_entity_boost_a030_learned_root_blend05_20260419_044815Z` | **0.41** | **0.11** | `branch_cap` 25 / `minrel` 5 | **未优于**烟测锚；**过门但边际弱** → **满 500 优先度低于 `max_evidence` 臂** |

**ctx-gold（辅助，口头一带）**：`cosine_probe` 臂 **`frac_samples_with_any_gold_in_context` 达 0.795**，与 **`never_visit` 大降** 同向，但 **accept 侧变差**，印证 **「看见 ≠ 进上下文/被接受」** 需分列讨论。

---

## 5. 瓶颈与解释（导师常问）

1. **Oracle 上界与真实导航仍有 gap**：说明系统瓶颈仍在 **导航与证据消费链路**（发现、排序、预算、接受），而非单靠「换更大生成器」一句话可解。  
2. **`never_visit` 压下来之后，剩余矛盾常落在 `visit_miss` / `gold_miss_evi`**：即 **「visit 过金叶仍未进 accept / context」**；与 **「从未 visit 金叶」** 不应混成一条叙事。  
3. **单变量必须同 manifest、同训练 checkpoint（learned 臂）**：否则 Δ 不可比；**`learned_root` checkpoint 为训练产物**，仓库不强制附带，运行前须存在 **`router_checkpoint_path`** 指向的文件（见 **`Navigation_Experiment_Record_CN.md` §6.5**）。

---

## 6. 下一步计划（可写成「下周交付物」）

| 优先级 | 动作 | 验收物 |
|:---:|:---|:---|
| P0 | **`max_evidence=14` 导航满 500**（无 `--max-samples`） | 新 `batch_id` + `accept_gate_audit` 摘要；与 **`122155Z`** 对照 **`never_visit` / `visit_miss` / ctx-gold**；劣化 ≥2pp 则熔断 |
| P1 | **`122155Z` 同旋钮端到端（e2e）** 复验 | 过程指标与 **EM/F1** 是否**同向**（避免仅凭导航表改主叙事） |
| P2 | **`cosine_probe` / `learned_root`** | **仅在**过程门满足或另有 **e2e** 动机时升格；当前 **`cosine`** 以 **trade-off** 归档 |

---

## 7. 口头汇报建议顺序（约 15 分钟）

1. **30 s**：一句话课题 + 「本阶段先闭环、再传导」。  
2. **2 min**：四模块结构（上一节表格）。  
3. **4 min**：**§4.1** 主锚（`041200Z` → `122155Z`），强调 **可审计、满量口径**。  
4. **3 min**：**§4.2** 三条 `n=200` ④，强调 **硬门判停**（`cosine` 不过门）。  
5. **2 min**：瓶颈（§5）+ 下一步（§6）。  
6. **1 min**：风险与诚实边界（不夸大、Oracle 不混用）。  
7. 留 **2～3 min** Q&A。

---

## 8. 附录（备查，口头可不说）

- **主要实验记录**：`docs/research/Navigation_Experiment_Record_CN.md`（**§6.0、§6.6、§6.7**）。  
- **审计 JSON 路径示例**：`outputs/reports/accept_gate_audit_<batch_id>.json`。  
- **仓库同步**：以当时服务器 **`git rev-parse HEAD`** 为准（例如与 **`origin/main`** 对齐的提交）。

---

*本文档由仓库内事实与既有记录整理而成，用于组会/导师汇报；若与 **`Navigation_Experiment_Record_CN.md`** 冲突，**以实验记录专档为准**。*
