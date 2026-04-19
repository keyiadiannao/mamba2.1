# Navigation 实验记录（阶段一）

## 1. 文档目的与边界

固定 **导航相关** 实验事实（`batch_id`、模版路径、关键表），避免结果散在终端。**不**承担完整论文叙事：判停细则、MI 口径、第二阶段总叙事以 **`Major_Issues_And_Resolutions_CN.md`**、**`SSGS_Research_Framework_CN.md`** 为准（框架文档 **§15～§16** 为阶段叙事缩略，**执行顺序与 `batch_id` 以本文件 §6 为准**）；本文件 **§9 起** 多为 **历史表归档**；**当前默认旋钮与近期单变量结论** 优先看 **§6.0、§6.5～6.7**。

---

## 2. 阶段背景（压缩）

- **目标**：真实子集上稳定跑通 **`370M` + 轻量 routing**；主对比 **`rule` / `cosine_probe`**（**`learned_classifier` 全树** 作补充/负结果，不进主表）。  
- **数据**：`2WikiMultiHopQA` → `extract_2wiki_subset` → `prepare_2wiki_subset` → `prepare_wiki_longdoc_subset` → `build_navigation_inputs_from_jsonl`（树 + manifest）。  
- **工程**：`Mamba` smoke/`370m hf_pretrained`、批内 **共享 Controller**、`run_registry` / `navigation_summary`、服务器可跑通预处理链。

---

## 6. 关键实验记录

### 6.0 主表、烟测与「是否还要 n=200」

- **主表与默认旋钮**（与 Oracle gap、是否改 **`122155Z`** 等）：**只认满 `manifest`**。当前 **`real_corpus_navigation_batch.json` 为 500 条** → **`run_navigation_batch.py` 不写 `--max-samples`**（或显式 **`--max-samples 500`**）跑出的 **`batch_id`** 与 **`accept_gate_audit`** 摘要。  
- **`--max-samples 200` 烟测仍保留**，但角色是 **辅助**，不是第二套主表：**(1) 熔断**——过程门（如 **`visit_miss`≤0.12**）、**`never_visit` 劣化 ≥2pp** 等可先在前 200 条判停，省满量 GPU；**(2) 对齐**——新模版先与历史 **`114138Z`** 等同切片对表，再决定是否上 500；**(3) 归因**——如 **`summarize_never_visit_nav_signals.py`** 在 200 上已够读 **`event_log`**。  
- **同一旋钮、同一协议** 若 **已有满 500 行**（例：**`122155Z`** vs **`162118Z`** 的 **`probe_budget`**）：**主叙事只写 500**；`n=200` **不必重复长解读**，表中可并成 **「与满量同向 / 仅作早停序」** 一句。  
- **`n≤50` / `n=10`**：仅 **链路、CI、脚本冒烟**；**不与 500 主结论同级**（见 §6.6 **P0-A′ 导航烟测 `n=10`** 读法）。  
- **与 §9 互斥读法**：**「下一刀 / 执行顺序」** 以 **§6.5～6.7** 及紧随其后的 **`bash` 复现块** 为准；**§9 起** 为 **历史归档**，若叙述与 §6 冲突 **以 §6 为准**。
- **与框架 / 阶段说明的叙事对齐（P0 诊断 vs P1 路径递归）**：本文件中的 **`122155Z` / `151107Z`** 等满量导航批，默认属于 **`SSGS_Research_Framework_CN.md` §5.0** 所定义的 **P0 诊断协议** 下的事实；**「句向量诊断」** 等表的结论域是 **「同一 SSGS/Router/融合下的编码器可替换性」**，**不**自动外推到 **P1 路径递归协议**（该协议须 **独立 `batch_id` 前缀与文档小节**，见 **`docs/meetings/First_Progress_Report_CN.md` §3**）。**勿与本文件 §6.6 历史标签「P1 读侧」**（`context_select`）**混用同一缩写语义**。

### 6.1 20 条真实子集

用途：

- 验证真实子集进入正式实验的第一跳是否稳定

主要观察：

- `370M + rule` 与 `370M + cosine_probe` 都已跑通
- routing 差异开始出现
- 但样本量仍偏小，不足以作为当前主结果规模

### 6.2 50 条真实子集

用途：

- 观察 routing 差异是否稳定

主要观察：

- `nav_success_rate = 1.0`
- rollback 行为开始明显出现
- learned classifier 已接入，但表现明显不稳定

### 6.3 100 条真实子集

用途：

- 检查轻量 routing 差异是否继续存在

主要观察：

- `rule` 与 `cosine_probe` 的时间差方向出现过波动
- 但 rollback 行为差异仍然稳定存在
- `avg_evidence_count` 多次打满 evidence budget，说明 evidence 数量暂不适合作为主结论

### 6.4 500 条真实子集（当前主结果）

当前建议固定为阶段一主结果的实验规模。

关键结果：

1. `370M + rule`
- `nav_success_rate = 1.0`
- `avg_nav_wall_time_ms ≈ 1147.35`
- `avg_rollback_count ≈ 1.082`
- `avg_evidence_count = 8.0`

2. `370M + cosine_probe`
- `nav_success_rate = 1.0`
- `avg_nav_wall_time_ms ≈ 1212.99`
- `avg_rollback_count ≈ 0.458`
- `avg_evidence_count = 8.0`

当前可稳说的 measured result：

- 两种 routing 都能在 `500` 条真实子集上稳定完成导航
- `rule` 更倾向于激进搜索，回溯更多
- `cosine_probe` 更倾向于保守搜索，回溯更少
- 两者耗时处于同一量级

当前不可过度宣称的点：

- 不应把 evidence 数量差异写成主结论
- 不应把“谁绝对更快”写成过强结论
- 不应把 learned classifier 当前结果写成主线收益

### 6.5 学习式 root + rule 混合（`learned_root_blend_alpha`，2026-04）

**目的**：在冻结启发式协议（`cos0.7 + probe1 + e8 + pool20`）下，验证 **线性 root 头** 在真实语料上是否可用；结论为 **纯 learned 不可用，必须与 `RuleRouter` 分数混合**。

**训练数据口径（与扫参对照无关，须固定）**：

- 导出：`scripts/run_nav/export_router_training_data.py` 的 **`--root-only`** + **`--max-root-children 128`**（示例输出 `router_training_data_root_v2.jsonl`）。
- 训练：`scripts/run_nav/train_learned_router.py`，**`--loss listwise_softmax`**（默认）。**约定输出路径**：**`configs/router/learned_root_router_real_corpus.json`** — 与各 **`learned_root`** 实验 JSON 里的 **`router_checkpoint_path`** 一致，便于换机不报路径错。  
  - **仓库里通常没有该文件**：Git 里只跟踪轻量示例 **`configs/router/learned_router_demo.json`**（脚本 **`--output` 默认值**也是它）；**真实语料 checkpoint 须本机训练生成**（例如对导出好的 jsonl 执行 **`train_learned_router.py --input <你的jsonl> --output configs/router/learned_root_router_real_corpus.json`**），或把已有权重 JSON **复制/改名**到上述路径，或**直接改**配置里的 **`router_checkpoint_path`** 指向你机器上的文件。  
  - 若某处写「需仓库里存在」**应理解为**：跑 **`learned_root`** 相关批前，**运行环境中**须能 `open` 到 **`router_checkpoint_path`** 指向的文件；**不是**要求该权重已提交到 Git。
- **扫 `α` 或换 rule 对照臂时，不要改上述 jsonl / checkpoint**，除非明确在做「新训练数据消融」；否则只改导航配置里的 **`learned_root_blend_alpha`** 与 **`batch_id_prefix`**。

**主结果批（`N=500`，`real_corpus_navigation_batch.json`）**：

| 项 | `α=0.25` | `α=0.5`（**仓库默认**，2026-04-17 500 复验） |
|:---|:---|:---|
| `batch_id` | `nav_real_corpus_mamba370m_learned_root_v2_cap128_train_20260417_131511Z` | `nav_real_corpus_mamba370m_learned_root_blend_a0_5_500_20260417_144435Z_20260417_144435Z` |
| `nav_success_rate` | `1.0` | `1.0` |
| `exact_match_rate` | `0.11` | `0.112`（同量级） |
| `avg_nav_wall_time_ms` | `≈1913` | **`≈1352`**（明显更快） |
| `analyze_evidence_saturation` | `visited≈0.456`，`accepted≈0.392`，`gold_missing=304` | `visited≈0.454`，`accepted≈0.394`，`gold_missing=303`（同量级） |
| 全量金叶报告 json（示例） | — | `outputs/reports/evidence_sat_blend_a0_5_500_144435Z.json` |

**默认 `α` 选择**：金叶过程指标在 **500** 上 **`0.25` 与 `0.5` 无实质差异**；**`0.5` 在 `nav_ms` 上显著更优**、EM 不降，故 **工程默认改为 `learned_root_blend_alpha=0.5`**（未写配置键时 `factory` 回退默认亦为 `0.5`）。若需复现 **`0.25`** 行为，在 JSON 中显式写 **`0.25`**。

**关于 `α > 0.5`（维护决策）**：**`α→1` 已证为金叶灾难区**，`0.5` 与 `0.25` 在金叶上已贴顶同量级，**不再把继续调高 `α`（如 0.6、0.7）列为常规实验必做项**；算力优先转向 **训练分布 / 端到端 / 其它瓶颈**。若未来仅为 **论文或安全上界** 需要，可 **烟测 `0.6`～`0.7` → 金叶硬闸门 → 再决定是否上 500**，且 **一旦出现金叶明显劣于 `0.5` 即停**。

**对照锚点**：同一协议下 **纯 `learned_root_classifier`（`learned_root_blend_alpha=1` 或未实现混合前的行为）** 在 `500` 上金叶访问近零（`frac_gold_leaf_ever_visited_deduped` 约 `0.002` 量级），与上表差异来自 **混合而非再训练**。

**小样本扫参（与全量同一 manifest、同一 checkpoint）**：

- 使用 `scripts/run_nav/run_navigation_batch.py` 的 **`--max-samples N`**（仅截取 manifest 前 `N` 条，**不修改** manifest 文件）。
- 建议 `N=50`；`α ∈ {0, 0.25, 0.5}` + **rule 冻结基线** 各跑一批，`batch_id_prefix` 带时间/后缀便于检索。
- **不必依赖独立 `.sh`**：在终端用短 **`python -c`** 写出临时配置再跑即可（见下）；跑完可删 `/tmp/nav_smoke_*.json`，避免配置碎片堆积。

**终端粘贴示例（仓库根目录、`N=50`；每段跑完再贴下一段；`STAMP` 自行改成固定字符串亦可）**：

```bash
cd /root/autodl-tmp/mamba2.1
export STAMP=$(date -u +%Y%m%d_%H%M%SZ)
export NAV_SMOKE_N=50
```

Rule 冻结基线一批：

```bash
python <<'PY'
import json
from pathlib import Path
import os
n, stamp = int(os.environ["NAV_SMOKE_N"]), os.environ["STAMP"]
c = json.loads(Path("configs/experiment/navigation_batch_real_corpus_rule_frozen_heuristic_cos07_probe1_e8_pool20.example.json").read_text(encoding="utf-8"))
c["batch_id_prefix"] = c["run_id_prefix"] = f"nav_smoke{n}_rule_{stamp}"
Path("/tmp/nav_smoke_rule.json").write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
python scripts/run_nav/run_navigation_batch.py --config /tmp/nav_smoke_rule.json --max-samples "$NAV_SMOKE_N"
```

Learned 一批（改 **`BLEND_ALPHA`** 为 `0.0`、`0.25`、`0.5` 各跑一次；若需更细网格可再加 **`0.1`、`0.3`** 等）：

```bash
export BLEND_ALPHA=0.25
python <<'PY'
import json
from pathlib import Path
import os
n, stamp = int(os.environ["NAV_SMOKE_N"]), os.environ["STAMP"]
a = float(os.environ["BLEND_ALPHA"])
c = json.loads(Path("configs/experiment/navigation_batch_real_corpus_learned_root_v2.example.json").read_text(encoding="utf-8"))
c["learned_root_blend_alpha"] = a
tag = str(a).replace(".", "_")
c["batch_id_prefix"] = c["run_id_prefix"] = f"nav_smoke{n}_blend_a{tag}_{stamp}"
Path("/tmp/nav_smoke_learned.json").write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
python scripts/run_nav/run_navigation_batch.py --config /tmp/nav_smoke_learned.json --max-samples "$NAV_SMOKE_N"
```

```bash
rm -f /tmp/nav_smoke_rule.json /tmp/nav_smoke_learned.json
```

**台账：`N=50`、blend `α` 小网格（2026-04-17，服务器）**  
说明：`batch_id` 由配置里的 `batch_id_prefix` 与 `make_run_id` 追加时间戳拼接而成，故出现 **两段日期后缀** 属正常。

| `learned_root_blend_alpha` | `batch_id`（完整字符串） |
|---:|:---|
| `0.0` | `nav_smoke50_blend_a0_0_20260417_141228Z_20260417_142316Z` |
| `0.1` | `nav_smoke50_blend_a0_1_20260417_141228Z_20260417_143013Z` |
| `0.25` | `nav_smoke50_blend_a0_25_20260417_141228Z_20260417_142547Z` |
| `0.3` | `nav_smoke50_blend_a0_3_20260417_141228Z_20260417_143159Z` |
| `0.5` | `nav_smoke50_blend_a0_5_20260417_141228Z_20260417_142841Z` |

过程指标请用 `scripts/diagnostics/analyze_evidence_saturation.py --batch-id '<上表>'` 自行汇总；**`N=50` 烟测**通过后已在 **`N=500`** 上复验 **`α=0.5`**（见上表）。

### 6.6 维护台账与下一阶段（2026-04，精简要）

**别混三件事**：① **导航金叶**（visited / accepted / `gold_missing`）由 **`SSGSController` + Router**；② **`context_select_*`** 只重排 **已访问叶 → context**，P1 已实证 **pool / k 不改金叶**，`entity_match` 仅 **2/200** 检索 EM 波动；③ **`oracle_item_leaves`** = 金叶直灌 context 的 **上界**，非可部署导航；**已有 Oracle 500 e2e 时不必再跑** 导航批 Oracle 200（除非写附录同切片表）。**`α>0.5` 不常规化**（§6.5 末）。

**P0 端到端（`500`，已完成）** — 模版：`end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav.example.json` / `…p0_learned_root_blend05.example.json`。脚本：`run_end_to_end_batch.py`（建议先 `--max-samples 10`）。摘要：`outputs/reports/end_to_end_batches/<batch_id>/batch_summary.json`。金叶 / ctx-gold：`analyze_evidence_saturation.py --registry-jsonl outputs/reports/run_registry.jsonl --batch-id '<id>'`（可加 `--with-context-gold-metrics`）。**`EM` 全零**：manifest **`reference_answer` 为 list** → **`normalize_reference_for_scoring`**（**MI-008**）。

**仓库默认（2026-04-18）**：**`…p0_rule_frozen_nav.example.json`** 与 **`navigation_batch_real_corpus_p0_frozen_nav_reg200_rule.example.json`** 已 **`explore_root_probe_budget_per_child=2`**（与已验证 e2e **`…probe_budget2_20260418_060702Z`** 一致）。下表 **A `rule`** 行仍为 **历史 `probe1`** 批（`154358Z`）；**刷新主表**须用默认模版 **重跑满 500** 得新 **`batch_id`**（可与 **`060702Z`** 并列作「默认=probe2」记录）。

| 臂 | `batch_id` | EM | F1 | `nav_ms` |
|:---|:---|---:|---:|---:|
| A `rule` frozen | `end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_20260417_154358Z` | **0.186** | **≈0.205** | **≈1247** |
| B `learned_root` `α=0.5` | `end_to_end_p0_real_corpus_370m_qwen7b_learned_root_blend05_20260417_160609Z` | **0.200** | **≈0.221** | **≈1301** |

**Accept 门审计（`audit_accept_gate.py`，与上表同一两 `batch_id`，`n=500`）** — 见 `scripts/diagnostics/audit_accept_gate.py`。指标含义：`frac_samples_never_visit_any_gold` = 有金标注但 **从未 visit 任一金叶** 的样本占比；`frac_samples_visit_gold_but_missing_accept_for_some_visited_gold` = **visit 到金叶但至少有一片已 visit 金叶未进 `accept_evidence`** 的样本占比；`visited_not_accepted_dispositions_aggregated` = 未 accept 金叶次数按 **`reject_leaf`（阈值）/ `reject_leaf_branch_cap`（配额）** 粗归因。

| 臂 | `frac_samples_never_visit_any_gold` | `frac_samples_visit_gold_but_missing_accept…` | `sum_gold_leaves_never_visited` / `…_visited_not_accepted` | `reject_leaf_branch_cap` / `reject_leaf_min_relevance`（叶次） |
|:---|:---:|:---:|:---:|:---:|
| A `rule` frozen | **0.572** | **0.16** | **1432** / **118** | **85** / **33** |
| B `learned_root` `α=0.5` | **0.546** | **0.154** | **1427** / **109** | **76** / **33** |

**读法**：**主矛盾仍是「根本 visit 不到金叶」**（~55%～57% 样本）；**Accept 门**在「已 visit 金叶却未进槽」子问题上约占 **15%～16% 样本**，叶次上 **`reject_leaf_branch_cap` 多于 `min_relevance`**，调 **`evidence_max_per_root_child` / explore 预算** 可能比单降 **`min_relevance_score`** 更对症，但仍须 **单变量** 烟测。**B 臂**相对 A：**未 visit 样本占比略降**、**分支 cap 挡掉的叶次略少**，与混合 root 改善 visit 的叙事一致。

**P0-2 导航批 `N=200`（已完成）** — 模版：`navigation_batch_real_corpus_p0_frozen_nav_reg200_{rule,learned_root_blend05}.example.json`；`run_navigation_batch.py --max-samples 200`。

| 臂 | `batch_id` | visited | `gold_missing` | 检索 EM | `nav_ms` |
|:---|:---|:---:|:---:|:---:|:---:|
| A `rule` | `nav_p0_reg200_rule_frozen_20260418_014016Z` | 0.41 | 130 | 0.11 | ≈1363 |
| B blend 0.5 | `nav_p0_reg200_learned_root_blend05_20260418_014536Z` | 0.445 | 122 | 0.125 | ≈1353 |

（accepted、饱和等见各批 `analyze_evidence_saturation` 落盘 json。）

**P1 读侧（`rule`、`N=200`，已收口）**

| 项 | `batch_id` | 改动 | 金叶 vs P0-2 rule | 检索 EM |
|:---|:---|:---|:---|:---|
| P1-1 pool | `nav_p1_reg200_rule_pool32_20260418_022308Z` | pool 32 | 同 | 同 |
| P1-2 k | `nav_p1_reg200_rule_overlap_k5_20260418_023920Z` | k 5 | 同 | 同 |
| P1-3 mode | `nav_p1_reg200_rule_entity_match_k4_20260418_030137Z` | entity_match | 同 | 0.12（+2/200） |

配置目录名：`…p1_rule_frozen_nav_reg200_pool32` / `…overlap_k5` / `…entity_match_k4`。

**Oracle 导航批 200（附录）**：`navigation_batch_real_corpus_nav_reg200_oracle_item_leaves.example.json`。

**P0-A′（单变量）**：基线 P0 两臂上 **`evidence_max_per_root_child` 缺省为 0**、**`reject_leaf_branch_cap` 主要来自根探测 Phase1 的 `root_branch_budget`**（实现见 `src/controller/ssgs_controller.py`：`explore_root_probe_budget_per_child` → `cap_source=top_m_budget`）。**第一刀**仅把 **`explore_root_probe_budget_per_child`：`1` → `2`**，其余与 P0 模版一致，避免与 `min_relevance_score` 混变量。

- **仅导航批（推荐先做：不加载 7B、无端到端 EM）**：`run_navigation_sample` 默认 **`run_generator=false`**，只跑 Mamba 导航 + trace。模版：  
  `configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_rule.example.json`、  
  `…p0_probe_budget2_learned_root_blend05.example.json`；**严对照 `probe1` 满 500**：  
  `navigation_batch_real_corpus_p0_probe_budget1_rule.example.json`、  
  `…p0_probe_budget1_learned_root_blend05.example.json`。  
  脚本：`python scripts/run_nav/run_navigation_batch.py --config '<上列之一>'`；与 P0-2 对齐可先 **`--max-samples 200`**；要与端到端主表对齐则 **不写 `--max-samples`**（跑满 **`samples_path` manifest**，当前 **`real_corpus_navigation_batch.json` 为 500 条** 即 **`n=500`**）。**不需要**设置 `GENERATOR_HF_MODEL_NAME`。  
  **AutoDL 示例（整段复制；请只用一行 `cd`，不要带 `|| exit 1`，否则失败时当前 shell 会直接退出）**：

```bash
conda activate mamba2
cd ~/autodl-tmp/mamba2.1
git pull origin main
python scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_rule.example.json \
  --max-samples 10
python scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_learned_root_blend05.example.json \
  --max-samples 10
```

**扩大样本（二选一）**：每次跑批都会生成**新的** `batch_id`（终端 JSON 与 `outputs/reports/batches/<batch_id>/batch_summary.json`）。下面诊断里的 **`BATCH_ID_RULE` / `BATCH_ID_LEARNED`** 必须改成你刚跑完打印的两串完整 id。

```bash
conda activate mamba2
cd ~/autodl-tmp/mamba2.1
git pull origin main

python scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_rule.example.json \
  --max-samples 200

python scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p0_probe_budget2_learned_root_blend05.example.json \
  --max-samples 200

BATCH_ID_RULE='nav_p0_probe_budget2_rule_YYYYMMDD_HHMMSSZ'
BATCH_ID_LEARNED='nav_p0_probe_budget2_learned_root_blend05_YYYYMMDD_HHMMSSZ'

for id in "$BATCH_ID_RULE" "$BATCH_ID_LEARNED"; do
  python scripts/diagnostics/analyze_evidence_saturation.py \
    --registry-jsonl outputs/reports/run_registry.jsonl \
    --batch-id "$id" \
    --out-json "outputs/reports/evidence_saturation_${id}.json"
  python scripts/diagnostics/audit_accept_gate.py \
    --registry-jsonl outputs/reports/run_registry.jsonl \
    --batch-id "$id" --root . \
    --out-json "outputs/reports/accept_gate_audit_${id}.json"
done
```

**`batch_id` 免手抄**：`run_navigation_batch.py` / `run_end_to_end_batch.py` 结束时会打印 **`__SSGS_BATCH_ID__=<id>`**；`audit_accept_gate.py` **须加 `--root .`**（并建议 **`--out-json`**），否则 **`context_*` 金叶列**为空；多份报告可用 **`scripts/diagnostics/print_diagnostic_summaries.py`** 终端打表。

**满 manifest（与端到端主表同条数；当前 manifest 为 500）**：上面两条 `run_navigation_batch.py` **不要加** **`--max-samples`**（或显式 **`--max-samples 500`** 与 manifest 一致）。**P0-A′ `probe_budget2` 满量台账**即按此协议在 AutoDL 跑出（见下表 `sample_count=500`）。

**P0-A′ 导航烟测（`n=10`，2026-04-18，AutoDL）** — 仅验证脚本与注册表；**方差极大，不得写入与 P0-2 / 500 主表同级结论**。

| 臂 | `batch_id` | `analyze_evidence_saturation`（摘录） | `audit_accept_gate`（摘录） |
|:---|:---|:---|:---|
| `probe_budget2` `rule` | `nav_p0_probe_budget2_rule_20260418_040256Z` | `frac_gold_leaf_ever_visited_deduped=0.2`，`frac_gold_in_accepted_evidence=0.2`，`gold_missing_from_evidence=8/10`，`frac_evidence_budget_saturated=1.0` | `frac_samples_never_visit_any_gold=0.8`，`sum_gold_leaves_visited_not_accepted=0`，`dispositions` **空** |
| `probe_budget2` `learned_root` `α=0.5` | `nav_p0_probe_budget2_learned_root_blend05_20260418_040319Z` | `frac_gold_leaf_ever_visited_deduped=0.1`，`frac_gold_in_accepted_evidence=0.1`，`gold_missing_from_evidence=9/10`，`frac_evidence_budget_saturated=1.0` | `frac_samples_never_visit_any_gold=0.9`，`sum_gold_leaves_visited_not_accepted=0`，`dispositions` **空** |

**读法（烟测尺度）**：**learned** 上 **`never_visit_any_gold=0.9`** 在 **10 条**上噪声极大，**不能**外推到 500；**`visited_not_accepted=0` + 空 dispositions** 只说明「这一小撮里几乎没人 visit 到金叶或 visit 后都 accept 了」，**不能**推出「P0 上 `reject_leaf_branch_cap` 已消失」——基线 500 审计里 cap 仍显著，须 **`n≥200` 或满 manifest** 再对 **`reject_leaf_branch_cap` 叶次**下判断。**rule** 侧 **`gold_missing=8/10`** 与 **`visited_deduped=0.2`** 同向，仅作链路检查。

**P0-A′ 导航满 manifest（`explore_root_probe_budget_per_child=2`，2026-04-18，AutoDL；本 manifest `sample_count=500`）** — `run_navigation_batch.py`（**无** `--max-samples`）+ 上列诊断；落盘 **`outputs/reports/evidence_saturation_nav_p0_probe_budget2_rule_20260418_041200Z.json`**、**`…learned_root_blend05_20260418_042544Z.json`**。

| 臂 | `batch_id` | `frac_gold_leaf_ever_visited_deduped` | `frac_gold_in_accepted_evidence` | `gold_missing`（条） | `audit`：`never_visit_any_gold` | `visit…missing_accept` | `sum_…_visited_not_accepted` | `reject_leaf_branch_cap` / `reject_leaf_min_relevance`（叶次） |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `probe_budget2` `rule` | `nav_p0_probe_budget2_rule_20260418_041200Z` | **0.42** | **0.388** | **306** | **0.58** | **0.116** | **75** | **44** / **31** |
| `probe_budget2` `learned_root` `α=0.5` | `nav_p0_probe_budget2_learned_root_blend05_20260418_042544Z` | **0.45** | **0.418** | **291** | **0.55** | **0.116** | **73** | **41** / **32** |

**复跑（仍为 `probe_budget=2`、`rule`、满 500，不是 `probe1`）**：`batch_id=nav_p0_probe_budget2_rule_20260418_045515Z` — 摘要与上表 **rule** 行（`041200Z`）一致；落盘 **`outputs/reports/evidence_saturation_nav_p0_probe_budget2_rule_20260418_045515Z.json`**。

**P0-A′ 严对照：导航满 500、`probe_budget=1`（已完成，2026-04-18）** — 模版：`navigation_batch_real_corpus_p0_probe_budget1_rule.example.json`、**`…learned_root_blend05.example.json`**；落盘 **`outputs/reports/evidence_saturation_nav_p0_probe_budget1_rule_20260418_051729Z.json`**、**`…learned_root_blend05_20260418_053127Z.json`**。

| 臂 | `batch_id` | `frac_gold_leaf_ever_visited_deduped` | `frac_gold_in_accepted_evidence` | `gold_missing`（条） | `audit`：`never_visit_any_gold` | `visit…missing_accept` | `sum_…_visited_not_accepted` | `reject_leaf_branch_cap` / `reject_leaf_min_relevance`（叶次） |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| `probe_budget1` `rule` | `nav_p0_probe_budget1_rule_20260418_051729Z` | **0.428** | **0.364** | **318** | **0.572** | **0.16** | **118** | **85** / **33** |
| `probe_budget1` `learned_root` `α=0.5` | `nav_p0_probe_budget1_learned_root_blend05_20260418_053127Z` | **0.454** | **0.394** | **303** | **0.546** | **0.154** | **109** | **76** / **33** |

**与 §6.6 P0 端到端 500 `audit_accept_gate`（`probe_budget=1`）**：上表 **`audit`** 列与 **端到端表** 两 `batch_id`（`end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_20260417_154358Z`、`end_to_end_p0_real_corpus_370m_qwen7b_learned_root_blend05_20260417_160609Z`）的 **`audit` 摘要逐字段一致**，说明 **e2e 批**与 **纯导航、同 `probe_budget`** 在 accept 统计上对齐；此前 **「`probe2` 导航 vs `probe1` e2e」** 的对比混了 **probe 参数**，应以 **下表「仅差 `probe_budget`」** 为准。

**导航内 `probe_budget`：`1` → `2`（满 500，同臂对照）**

| 臂 | 指标 | `probe1` | `probe2`（上表 `041200Z` / `042544Z`） | 粗读 |
|:---|:---|---:|---:|:---|
| `rule` | `reject_leaf_branch_cap`（叶次） | 85 | 44 | **cap 挡叶次约减半** |
| `rule` | `frac_samples_visit_gold_but_missing_accept…` | 0.16 | 0.116 | **accept 子问题减轻** |
| `rule` | `frac_gold_in_accepted_evidence` | 0.364 | 0.388 | **金进槽略升** |
| `rule` | `frac_samples_never_visit_any_gold` | 0.572 | 0.58 | **同量级**（主瓶颈仍在 visit） |
| `learned_root` | `reject_leaf_branch_cap`（叶次） | 76 | 41 | 同向 |
| `learned_root` | `frac_samples_never_visit_any_gold` | 0.546 | 0.55 | 同量级 |

**P0-A′ 端到端满 500、`probe_budget=2`（已完成，2026-04-18）** — 与 **本节 P0 主表** 两臂相比，仅 **`explore_root_probe_budget_per_child=2`**。落盘 ctx-gold：**`outputs/reports/evidence_saturation_end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_probe_budget2_20260418_060702Z_ctxgold.json`**、**`…learned_root_blend05_probe_budget2_20260418_062859Z_ctxgold.json`**。**EM/F1** 已合入下表（`batch_summary.json` 口径）；无 **`jq`** 时复现打印可用 **`python3`**：

```bash
cd ~/autodl-tmp/mamba2.1
for id in \
  end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_probe_budget2_20260418_060702Z \
  end_to_end_p0_real_corpus_370m_qwen7b_learned_root_blend05_probe_budget2_20260418_062859Z
do
  python3 -c "import json; p='outputs/reports/end_to_end_batches/${id}/batch_summary.json'; d=json.load(open(p)); print(json.dumps({k: d.get(k) for k in ('batch_id','exact_match_rate','avg_answer_f1','avg_rouge_l_f1','avg_nav_wall_time_ms')}, indent=2, ensure_ascii=False))"
done
```

| 臂 | `batch_id` | EM（`exact_match_rate`） | F1（`avg_answer_f1`） | `avg_nav_wall_time_ms` | 相对 P0 主表基线 |
|:---|:---|---:|---:|---:|:---|
| `rule` + `probe2` | `end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_probe_budget2_20260418_060702Z` | **`0.188`** | **`≈0.208`** | **≈1197** | 基线 EM **`0.186`**（**`+0.002`**），F1 **≈0.205** |
| `learned_root` `α=0.5` + `probe2` | `end_to_end_p0_real_corpus_370m_qwen7b_learned_root_blend05_probe_budget2_20260418_062859Z` | **`0.198`** | **`≈0.221`** | **≈1281** | 基线 EM **`0.200`**（**`−0.002`**），F1 **≈0.221** |

**读法（终点）**：**`rule`** 上 **`probe2` 终点略升**（EM **`+0.002`**、F1 **略升**），与 **ctx-gold / accept 过程改善**同向但幅度小；**`learned_root`** 上 **EM 微降 `0.002`**、F1 几乎贴平基线，**不宜**仅凭 accept 侧改动就改「主默认臂」叙事。**结论**：**`probe2` 可记为 `rule` 侧可选默认候选**（与 **MI-004/005** 不冲突：过程升、终点未跌）；**learned** 仍优先以 **`probe1` 基线**写主表，除非后续复跑确认 **`probe2` 稳定优**。**主线**仍为 **`never_visit` ~0.55～0.58**（`audit`），须 **路由/探索** 再开刀。

**过程与 `audit`（与导航满 500 `probe2` 同向）**：两臂 **`frac_gold_leaf_ever_visited_deduped` / `frac_gold_in_accepted_evidence` / `audit`** 与 **§6.6 导航 `probe2` 表**（`041200Z` / `042544Z`）一致量级；**`mean_frac_gold_leaf_texts_in_generator_context`**：`rule` **≈0.159**、`learned` **≈0.170**（相对 **§9.12** `overlap_k4` 批 **`≈0.127`**，**ctx-gold 均值抬升**）。**`audit`**：`never_visit` **0.58 / 0.55**，`branch_cap` **44 / 41**。

**P0-B′：`rule` + `entity_boost_alpha=0.05` 组合臂（导航、不跑 7B 生成；2026-04-18 冻结）** — 主模版：`configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a005.example.json`。单因子消融：`configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a005_abl_{probudget_1,minrel_1p0,maxev_8,ctx_overlap_k4,maxnodes_64}.example.json`（各相对主模版**仅改一项**，小样本对照）。**`α>0.15` 探索**：`a020` 烟测（`112919Z`，`n=200`）相对 **`0.15` 烟测**边际；**`a030` 满 500（`122155Z`）** 相对 **`105756Z`** 全面占优（见上表末行前一格）。模版：`navigation_batch_real_corpus_p0_visit_rule_entity_boost_{a020,a030}.example.json`。

| 批 | `n` | 检索 `EM` | `audit`：`never_visit` | `reject_leaf_branch_cap` / `reject_leaf_min_relevance`（叶次） | `mean_frac_gold_leaves_in_context` | `sum_accepted_gold_not_in_context` | 备注 |
|:---|---:|---:|---:|---:|---:|---:|:---|
| `nav_p0_visit_rule_entity_boost_a005_20260418_075320Z` | 500 | 0.116 | — | — | — | — | 旋钮旧版（`min_rel=1.0`、`max_evidence=8`、`overlap_topk` k=4 等） |
| `nav_p0_visit_rule_entity_boost_a005_20260418_081727Z` | 500 | **0.126** | **0.488** | **63 / 10** | **0.210** | **49** | **冻结主行（`α=0.05`）**（`min_rel=0.6`、`max_evidence=12`、`max_nodes=80`、`question_entity_match_topk` k=6、pool 24、`probe_budget=2`） |
| `nav_p0_visit_rule_entity_boost_a010_20260418_095212Z` | 500 | 见 `outputs/reports/batches/<id>/batch_summary.json` 中 `exact_match_rate` | **0.448** | 见 `accept_gate_audit_<id>.json` 内 `visited_not_accepted_dispositions_aggregated` | **0.227** | **51** | **`α=0.1` 满 500**：相对 `081727Z`，`never_visit` **−4.0pp**；`saturation`：`frac_gold_leaf_ever_visited_deduped` **0.552**（**+4.0pp**）、`frac_gold_in_accepted_evidence` **0.514**、`sample_count_gold_missing_from_evidence` **243**；`visit…missing_accept` **0.106**（≤0.12）→ 阶段目标 **B** 正式达成 |
| `nav_p0_visit_rule_entity_boost_a015_20260418_105756Z` | 500 | 见 `outputs/reports/batches/<id>/batch_summary.json` 中 `exact_match_rate` | **0.420** | 见 `accept_gate_audit_<id>.json` 内 `visited_not_accepted_dispositions_aggregated` | **0.238** | **57** | **`α=0.15` 满 500**：相对 `095212Z`，`never_visit` **−2.8pp**；`gold_visit_dedup` **0.580**（**+2.8pp**）、`frac_gold_in_accepted_evidence` **0.540**、`sample_count_gold_missing_from_evidence` **230**；`visit…missing_accept` **0.108**（≤0.12）→ **阶段 A（收工级）双阈同时满足**（`never_visit`≤0.42 且 `gold_visit_dedup`≥0.58） |
| `nav_p0_visit_rule_entity_boost_a030_20260418_122155Z` | 500 | 见 `outputs/reports/batches/<id>/batch_summary.json` 中 `exact_match_rate` | **0.378** | 见 `accept_gate_audit_<id>.json` 内 `visited_not_accepted_dispositions_aggregated` | **0.258** | **63** | **`α=0.3` 满 500**：相对 `105756Z`，`never_visit` **−4.2pp**、`gold_visit_dedup` **+4.2pp**（**0.622**）、`frac_gold_in_accepted_evidence` **0.578**、`sample_count_gold_missing_from_evidence` **211**；`visit…missing_accept` **0.110**（仍 **≤0.12**）→ **`rule` 实体偏置臂当前默认候选**（`105756Z` 可作保守对照） |
| `…085815Z` 及 `…_abl_*`（`20260418`） | 100 | — | ≈0.53 簇 | — | — | — | **`probudget_1` / `maxev_8` / `minrel_1.0` / `overlap_k4` 不占优**；`avg_nav_wall_time_ms` 满量约 **1660**（较 `075320Z` ↑） |

**读法**：相对 **`probe2` rule 满 500（`041200Z`，`never_visit≈0.58`）**，**`α=0.05` 冻结行 `081727Z`** 将 **`never_visit` 压到约 0.49** 量级；**`α=0.1` 满 500（`095212Z`）** 达 **阶段 B**；**`α=0.15` 满 500（`105756Z`）** 达 **阶段 A**；**`α=0.3` 满 500（`122155Z`）** 在 **过程仍 ≤0.12** 前提下 **全面优于 `105756Z`**（`never_visit` **≈0.38**、`gold_visit_dedup` **≈0.62**），**建议**将 **`α=0.3`** 记为 **`rule` 实体偏置臂当前默认候选**。**`learned_root` / e2e** 与检索 EM **分列**（叙事见 **MI-004/005**）。

**P0-B′ 导航扫参收口（`2026-04-18`）** — **默认工作点**：**`122155Z`**（`configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a030.example.json`），保守对照 **`105756Z`（`α=0.15`）**。**已冻结、不默认续扫**：**`α` 细网格**、**`context_select_*`**、**`probe_top_m=2`**（表 **§6.7 ①b**）、**`max_nodes`/`max_depth` 单变量**（§6.7 **②**）。**仍存优化空间**（`gold_miss_evi`、回溯拓扑、**`learned`/Router** 与 **e2e 终点**）→ **`SSGS_Research_Framework_CN.md`** / **MI-004/005**。**P0 `end_to_end_*`** 已 **`entity_boost_alpha=0.3`** 与 **`question_entity_match_topk` / `max_evidence=12` / `min_rel=0.6` / `max_nodes=80`**；**`batch_id_prefix` / `run_id_prefix`** 带 **`_visit_a030`** 与旧 **`overlap_topk`** 跑批区分。

### 6.7 近期单变量（`122155Z` 栈，2026-04-18）

与 **§6.0**：**主叙事以满 500 为准**；**`n=200` 仍要保留**，用于 **熔断 / 对齐切片 / 归因**；**`probe_budget=3` 已有满 500（`162118Z`）**，其 **`n=200`（`160226Z`）** 仅证明 **同向**，表中 **不再展开长段**。

| 编号 | 单变量 | `n` | 代表 `batch_id`（可检索后缀） | `never_visit`（桶≈） | `visit_miss` | 结论 |
|:---:|:---|---:|:---|---:|---:|:---|
| ①b | `probe_top_m: 1→2` | 200 | `…142055Z` | 0.40 | **0.13** | 过 **0.12** 门→不推进满 500；与 **`071832Z`（栈不同）`** 勿混；**P0-B-1** |
| ①a | `root_entity_zero_overlap_fallback_beta=0.07` | 200 | `…144709Z` | 0.40 | 0.10 | **`never_visit` 不动**；默认 **`beta=0`** |
| ② | `max_nodes 112` / `max_depth 10` | 200 | `…151646Z` / `…153821Z` | 0.40 | 0.10 | **结案**：硬顶非主因；**`summarize_never_visit_nav_signals.py`（`151646Z`）**：`max_nodes_reached`/`max_depth_reached` **0%**，`mean(evidence_texts)=12` |
| ③ | `probe_budget 2→3` | 200 | `…160226Z` | 0.415 | 0.065 | **trade-off**；**满 500 见下行** |
| ③ | 同上 | 500 | `…162118Z` | **0.384** vs `122155Z` **0.378** | **0.066** vs **~0.110** | **`branch_cap` 叶次 42**；**不替换默认 `probe_budget=2`**，可选 **B′** |
| ④ | `learned_root` α=0.5 / `cosine_probe` / `max_evidence=14` | 200 | 模版 **`…a030_learned_root_blend05`** / **`…a030_cosine_probe`** / **`…abl_maxev_14`** | 待定 | ≤0.12 硬门 | **逼近 Oracle** = 真实导航抬金叶过程指标；**禁** `oracle_item_leaves` 与注入 **`leaf_indices_required`**（§6.6「勿混三件事」） |
| ④′ | **`cosine_probe`**（**`a030` 实体偏置栈**，与 **`…a030_cosine_probe.example.json`** 一致） | **500** | **`nav_p0_visit_rule_entity_boost_a030_cosine_probe_20260419_081150Z`** | **`never_visit=0.214`** | **`visit_miss=0.19`** | **与 `122155Z` 同 manifest 对表（2026-04-19，导航 `batch_summary` + `accept_gate_audit`）**：**检索 `exact_match_rate`** **0.204** vs **0.14**；**`avg_nav_wall_time_ms`** **≈2428** vs **≈2367**。审计摘要：**`never_visit` 0.214 vs 0.378**，**`visit_miss` 0.19 vs 0.11**；**`frac_samples_with_any_gold_in_context`** **0.784 vs 0.622**，**`mean_frac_gold_leaves_in_context`** **≈0.360 vs 0.258**，**`sum_accepted_gold_not_in_context`** **84 vs 63**；叶次 **cap/minrel** **116/18 vs 70/10**。**读法**：**cosine 在检索 rubric 与 ctx-gold 上优于当前 `rule` 默认臂**，**accept/cap 压力更高**（与 **`visit_miss`↑** 一致）。**`122155Z` 审计**：**`accept_gate_audit_nav_p0_visit_rule_entity_boost_a030_20260418_122155Z.json`**；换机无该文件时用 **`audit_accept_gate.py --batch-id-substring 122155Z`** 再生（如 **`…_rule_122155Z_regen.json`**，摘要应一致）。**`--out-json` 须含 `.json` 后缀**。**e2e 满 500**：见下 **`④′-e2e`**（**已跑**：**`…cosine_probe…_20260419_094151Z`**）。**勿**与 §9.12 旧 **`context_select` 三连**混表。 |

**④′-e2e（与 `081150Z` / `122155Z` 同旋钮、`probe_budget=2`、满 manifest）**：模版 **`configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_cosine_probe_nav_probe_budget2_visit_a030.example.json`** — 与 **`end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav_probe_budget2.example.json`**（`run_id_prefix` 已含 **`visit_a030`**）除 **`routing_mode`：`cosine_probe` vs `rule`** 外一致；**7B 生成 EM/F1** 与 §6.7 导航批 **retrieval proxy** 分列写结论。

```bash
cd ~/autodl-tmp/mamba2.1 && conda activate mamba2
GEN=/root/autodl-tmp/models/Qwen2.5-7B-Instruct

python scripts/run_eval/run_end_to_end_batch.py \
  --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_cosine_probe_nav_probe_budget2_visit_a030.example.json \
  --generator-hf-model-name "$GEN" | tee /tmp/e2e_cosine_visit_a030_full500.log

# 从日志或 outputs/reports/end_to_end_batches/<batch_id>/batch_summary.json 取 batch_id 后：
# python scripts/diagnostics/audit_accept_gate.py \
#   --registry-jsonl outputs/reports/run_registry.jsonl \
#   --batch-id "<上一步 batch_id>" \
#   --out-json "outputs/reports/accept_gate_audit_<上一步 batch_id>.json"
```

**`④′-e2e` 已跑满 500（2026-04-19，`370m` + 7B）**：**`batch_id`**：**`end_to_end_p0_real_corpus_370m_qwen7b_cosine_probe_nav_probe_budget2_visit_a030_20260419_094151Z`**；**`batch_summary`**：**`outputs/reports/end_to_end_batches/end_to_end_p0_real_corpus_370m_qwen7b_cosine_probe_nav_probe_budget2_visit_a030_20260419_094151Z/batch_summary.json`**。

**打印终点（与 §6.6 表同一键）**：

```bash
cd ~/autodl-tmp/mamba2.1
python3 -c "import json; p='outputs/reports/end_to_end_batches/end_to_end_p0_real_corpus_370m_qwen7b_cosine_probe_nav_probe_budget2_visit_a030_20260419_094151Z/batch_summary.json'; d=json.load(open(p,encoding='utf-8')); print(json.dumps({k:d.get(k) for k in ('batch_id','sample_count','exact_match_rate','avg_answer_f1','avg_nav_wall_time_ms')}, indent=2, ensure_ascii=False))"
```

**对表（同 manifest、`probe_budget=2`、`visit_a030` 栈）**：**`rule`** 历史 **`end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_probe_budget2_20260418_060702Z`**（见上 **§6.6** 表）— 若数据或代码与当时不一致，请以 **同机当场重跑** **`p0_rule_frozen_nav_probe_budget2.example.json`** 的 **`batch_summary`** 为准。

**`④′-e2e` 终点（`batch_summary`，500，`eval_mode=generation`）**：

| 臂 | `batch_id`（后缀） | `exact_match_rate` | `avg_answer_f1` | `avg_nav_wall_time_ms` |
|:---|:---|---:|---:|---:|
| **`cosine_probe`** | **`…cosine_probe…_20260419_094151Z`** | **`0.236`** | **`≈0.263`** | **`≈2295`** |
| **`rule`**（历史对表） | **`…rule_frozen_nav_probe_budget2_20260418_060702Z`** | **`0.188`** | **`≈0.208`** | **`≈1197`** |

**Δ（cosine − rule）**：**EM `+0.048`（约 `+4.8pp`）**，**F1 约 `+0.055`**；**`avg_nav_wall_time_ms` 约 `+1100ms/条`**（量级与 **§9.12** cosine 略慢于 rule 同向，此处差更大，**并列报告时写清同机负载/是否同 CUDA 图**）。**读法**：在 **`visit_a030` + `entity_match` 栈** 下，**导航侧 proxy 的优势已传导到 7B 生成 EM**（与 **`081150Z`/`122155Z` 过程对表**同向）；**是否升格默认臂**仍建议结合 **e2e `accept_gate_audit`** 与 **成本**再定。

**可选 `accept_gate_audit`（e2e 批）**：

```bash
python scripts/diagnostics/audit_accept_gate.py \
  --registry-jsonl outputs/reports/run_registry.jsonl \
  --batch-id "end_to_end_p0_real_corpus_370m_qwen7b_cosine_probe_nav_probe_budget2_visit_a030_20260419_094151Z" \
  --out-json "outputs/reports/accept_gate_audit_end_to_end_p0_real_corpus_370m_qwen7b_cosine_probe_nav_probe_budget2_visit_a030_20260419_094151Z.json"
python scripts/diagnostics/summarize_audit_failure_buckets.py \
  "outputs/reports/accept_gate_audit_end_to_end_p0_real_corpus_370m_qwen7b_cosine_probe_nav_probe_budget2_visit_a030_20260419_094151Z.json"
```

**句向量诊断（Mamba vs MiniLM，同 `a030` 栈、同融合）**：默认句向量 **`sentence-transformers/all-MiniLM-L6-v2`**（384 维，英文子集上常用；多语料可改键 **`sentence_transformer_model_name`** 为 **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**）。依赖 **`pip install sentence-transformers`**。模版 **`configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a030_sentence_minilm.example.json`** 与 **`…p0_visit_rule_entity_boost_a030.example.json`** 除 **`navigator_type`** 与 **Mamba 专用键**外对齐；**`merge_path_summaries` 与 Mamba HF 路径一致**。烟测：

```bash
python scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a030_sentence_minilm.example.json \
  --max-samples 50
```

**满 500 主表（`sentence_transformer` / MiniLM，`a030`+`rule`，2026-04-19）**：**`batch_id`**：**`nav_p0_visit_rule_entity_boost_a030_sentence_minilm_20260419_151107Z`**；**`batch_summary`**：**`outputs/reports/batches/…151107Z/batch_summary.json`**；**审计**：**`accept_gate_audit_…151107Z.json`**。与 **`122155Z`（Mamba+`rule`，满 500）** **同 manifest、同旋钮**，可作 **编码器诊断主对表**。

| 指标（导航批，`n=500`） | **`151107Z`（MiniLM）** | **`122155Z`（Mamba）** | **读法** |
|:---|---:|---:|:---|
| **`exact_match_rate`**（retrieval proxy） | **0.146** | **0.14** | MiniLM **略高 `+0.006`**（**非**大差距；二者仍属同量级）。 |
| **`avg_answer_f1`** | **≈0.155** | **≈0.150**（`122155Z` `batch_summary`） | 同向略优。 |
| **`avg_nav_wall_time_ms`** | **≈422** | **≈2367** | MiniLM **显著更快**（报告写清 **同机、同 CUDA**）。 |
| **`nav_success_rate`** | **1.0** | **1.0** | — |
| **`never_visit`** | **0.362** | **0.378** | MiniLM **略优**（**−1.6pp**）。 |
| **`visit_miss`** | **0.158** | **~0.11** | MiniLM **更差**（**+4.8pp**）；**accept** 侧压力更大。 |
| **`frac_samples_with_any_gold_in_context`** | **0.638** | **0.622** | MiniLM **略优**。 |
| **`mean_frac_gold_leaves_in_context`** | **≈0.265** | **≈0.258** | 接近，MiniLM **略优**。 |
| **`sum_accepted_gold_not_in_context`** | **48** | **63** | MiniLM **更低**（同 **`n`** 下可并列）。 |
| **叶次 `cap` / `minrel`** | **55 / 51** | **70 / 10** | MiniLM **`minrel` 叶次占比更高**；Mamba 侧 **`cap` 为主** → 瓶颈形状不同。 |

**诊断结论（压缩）**：在 **「同融合 + 同 SSGS/`rule`/读侧」** 下，**未出现「Mamba 在检索 proxy 上碾压 MiniLM」**；**MiniLM 更快、部分过程指标略好**，但 **`visit_miss` 明显更差**。论文叙事应 **诚实写成交替优势 + 代价轴**，而非预设 **Mamba 不可替代**；若需 **「Mamba 专属增益」**，须再设计 **更贴 SSM 的路径输入/探针**（见 **§5.3** 与 **`④′`** 讨论）。

**P1 路径递归协议（路线 B）— 烟雾与满量**：与 **`SSGS_Research_Framework_CN.md` §5.0** 一致，**禁止与 P0 主表混读**。实现要点：**`navigator_path_recursive_prompt: true`** 时，每步将 **`[Q]` / `[PATH]`（祖先节点文本截断链）/ `[NODE]`（当前节点）** 拼成 **单次 HF 前向**（**`navigator_load_strategy` 须为 `hf_pretrained`**）；**不再**对该步做 **`merge_path_summaries` 与上一步 hidden 的向量融合**（路径信息进 **token 串**）。控制器经 **`DocumentTree.build_node_index`** 解析 **`state.path`** 传入 **`path_ancestor_nodes`**。键名：**`navigator_path_prompt_max_chars_per_segment`**、**`navigator_path_prompt_max_question_chars`**；长提示建议 **`navigator_max_tokens_per_node`** **≥768**（模版默认 **896**）。

模版：**`configs/experiment/navigation_batch_real_corpus_p1_path_recursive_visit_rule_entity_boost_a030.example.json`**（与 P0 **`…p0_visit_rule_entity_boost_a030.example.json`** 同 **`a030`+`rule`+visit**，仅 **`batch_id_prefix`/`run_id_prefix`** 与 **`navigator_*` 路径递归键** 不同）。**烟雾（GPU，建议先 20～50 条）**：

```bash
cd ~/autodl-tmp/mamba2.1
python3 scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p1_path_recursive_visit_rule_entity_boost_a030.example.json \
  --max-samples 20
```

**首批链路跑通（服务器，2026-04-19）**：**`batch_id`**：**`nav_p1_path_recursive_visit_rule_entity_boost_a030_20260419_161956Z`** → **`outputs/reports/batches/nav_p1_path_recursive_visit_rule_entity_boost_a030_20260419_161956Z/batch_summary.json`**。**`sample_count` / 指标以该 JSON 为准**（是否加 **`--max-samples`** 由你本地命令决定）；**`accept_gate_audit`** 用同 **`batch_id`** 生成后，再与 P0 **`122155Z`** **分列**对读。

```bash
python3 -c "import json; p='outputs/reports/batches/nav_p1_path_recursive_visit_rule_entity_boost_a030_20260419_161956Z/batch_summary.json'; d=json.load(open(p,encoding='utf-8')); print(json.dumps({k:d.get(k) for k in ('batch_id','sample_count','exact_match_rate','avg_answer_f1','avg_nav_wall_time_ms','nav_success_rate')}, indent=2, ensure_ascii=False))"
python scripts/diagnostics/audit_accept_gate.py \
  --registry-jsonl outputs/reports/run_registry.jsonl \
  --batch-id "nav_p1_path_recursive_visit_rule_entity_boost_a030_20260419_161956Z" \
  --out-json "outputs/reports/accept_gate_audit_nav_p1_path_recursive_visit_rule_entity_boost_a030_20260419_161956Z.json"
```

跑通后核对 **`batch_summary.json`** 内 **`config.navigator_path_recursive_prompt`** 与 **`nav_success_rate`**；再 **`audit_accept_gate.py`** 与 P0 同 **`122155Z`** 旋钮对表（**分列叙事**，勿宣称「同 batch 续跑」）。**句向量 P1 烟测**：同一 JSON 将 **`navigator_type`** 改为 **`sentence_transformer`** 并保留 **`navigator_path_recursive_prompt`** 即可（依赖 **`sentence-transformers`**）。

**烟测 `n=200`（`150150Z`）**：保留作 **熔断/对齐切片**；**勿与上表混为最终结论**。

**打印 `151107Z` `batch_summary`**：

```bash
cd ~/autodl-tmp/mamba2.1
python3 -c "import json; p='outputs/reports/batches/nav_p0_visit_rule_entity_boost_a030_sentence_minilm_20260419_151107Z/batch_summary.json'; d=json.load(open(p,encoding='utf-8')); print(json.dumps({k:d.get(k) for k in ('batch_id','sample_count','exact_match_rate','avg_answer_f1','avg_nav_wall_time_ms','nav_success_rate')}, indent=2, ensure_ascii=False))"
```

**`accept_gate_audit`（`151107Z`，勿漏 `.json`）**：

```bash
python scripts/diagnostics/audit_accept_gate.py \
  --registry-jsonl outputs/reports/run_registry.jsonl \
  --batch-id "nav_p0_visit_rule_entity_boost_a030_sentence_minilm_20260419_151107Z" \
  --out-json "outputs/reports/accept_gate_audit_nav_p0_visit_rule_entity_boost_a030_sentence_minilm_20260419_151107Z.json"
python scripts/diagnostics/summarize_audit_failure_buckets.py \
  "outputs/reports/accept_gate_audit_nav_p0_visit_rule_entity_boost_a030_sentence_minilm_20260419_151107Z.json"
```

**`branch_cap` 机制（与 P0-A′ 一致）**：**`evidence_max_per_root_child=0`** 时，**`reject_leaf_branch_cap`** 的 **`cap` 常来自** **`explore_root_probe_budget_per_child`**（`cap_source=top_m_budget`，见 **`src/controller/ssgs_controller.py`**）。

**`probe_budget=3`：满 500 导航 + 可选 e2e**（**导航满 500 已产出 `162118Z`**；下列命令供 **复现 / 二次跑 / 换机**）

```bash
cd ~/autodl-tmp/mamba2.1
conda activate mamba2

# 导航满 500（无 --max-samples）
python scripts/run_nav/run_navigation_batch.py \
  --config configs/experiment/navigation_batch_real_corpus_p0_visit_rule_entity_boost_a030_nav500_probe_budget3.example.json | tee /tmp/nav_probudget3_full500.log
BATCH_ID=$(grep '__SSGS_BATCH_ID__=' /tmp/nav_probudget3_full500.log | tail -1 | cut -d= -f2-)
python scripts/diagnostics/audit_accept_gate.py \
  --registry-jsonl outputs/reports/run_registry.jsonl \
  --batch-id "$BATCH_ID" \
  --out-json "outputs/reports/accept_gate_audit_${BATCH_ID}.json"
python scripts/diagnostics/summarize_audit_failure_buckets.py \
  "outputs/reports/accept_gate_audit_${BATCH_ID}.json"

# e2e 满 500（路径与 generator 按服务器实际修改）
python scripts/run_eval/run_end_to_end_batch.py \
  --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav_probe_budget3_visit_a030.example.json
```

**满 500 导航（`probe_budget=3`）**：**`162118Z`**（`nav_p0_visit_rule_entity_boost_a030_nav500_probe_budget3_20260418_162118Z`）— **`frac_samples_never_visit_any_gold=0.384`**（较 **`122155Z`** **0.378** 约 **+0.6pp**，**未达 ≥2pp** 熔断）、**`frac_samples_visit_gold_but_missing_accept…=0.066`**（较 **`122155Z`** **~0.110** 约 **−4.4pp**）；`visited_not_accepted_dispositions_aggregated`：**`reject_leaf_branch_cap=42` / `minrel=10`**（叶次）；**`frac_samples_with_any_gold_in_context=0.616`**，**`mean_frac_gold_leaves_in_context≈0.256`**，**`sum_accepted_gold_not_in_context=70`**。审计 JSON **`accept_gate_audit_…162118Z.json`**，分桶 CSV **`audit_sample_buckets_…162118Z.csv`**。**结论**：**满量 trade-off 与 `n=200`（`160226Z`）同向**——**`visit_miss` 显著改善**，**`never_visit` 略升但在熔断线内** → **不替换 `122155Z` 默认（`probe_budget=2`）**；若业务 **强偏 accept/context 金叶** 可将 **`probe_budget=3`** 记为 **并列 B′ 臂**，**须**与 **`122155Z` 同 manifest e2e 的 `exact_match_rate`** 对表后再写主叙事。上表 **④** 行已列 **`learned` / `cosine_probe` / `max_evidence=14`** 模版与硬门；**`learned`/`cosine`** 与 **`rule` 主默认分列**（**§6.5**、**MI-004/005**）。

**`never_visit≈0.38` 读法（压缩）**：相对 **`probe2≈0.58`** 已 **~20pp**，但 **38% 未触金叶** 不可单靠再抬 **`α`** 归零。**`gold_miss_evi`（visit 后仍丢）** 与 **`never_visit`** **分列**：前者常是 **「看见了拿不到」** → 优先 **accept / 证据链**；后者再分 **结构 vs 实体失配**（**`accept_gate_audit` 聚类 + `run_payload`**）。**§6.7 ②** 已证 **`max_nodes`/`max_depth` 硬顶** 非主因。

**阶段目标（建议稿，可改台账）** — **导航（无生成）**：以 **`081727Z`** 为锚。**A（收工级）**：`never_visit` **≤0.42** 且 **`frac_gold_leaf_ever_visited_deduped` ≥0.58**（主矛盾明显让位）。**B（本阶段合格）**：相对 `081727`，`never_visit` **再降 ≥3pp**（≤**0.458**）**或** `frac_gold_leaf_ever_visited_deduped` **再升 ≥3pp**，且 **`visit…missing_accept` 不高于 `0.12`**、**`reject_leaf_branch_cap` 叶次相对 `081727`（63）增幅 ≤15**（实体臂 cap 已高于纯 `probe2` 的 44，**不宜**再套用「≤49」旧阈）。**C（仅记阴性）**：`never_visit` 不降反升 **≥2pp** 或 cap **>80** → 停 **`α=0.15`**，转 **`max_nodes=96`** 或收口。**端到端（7B 生成 EM）**：同 manifest 上 **相对当前主表 `learned_root` ~0.20**，单轮导航改动 **≥+0.01 EM** 已属强信号；**≥+0.005** 可作「值得写进主表」下限；须与 **过程指标同向**（**MI-004/005**）。

**P0-B 扫参约束（可套用，精简）** — 相对导航基线 **`nav_p0_probe_budget2_rule_20260418_041200Z`**（**`never_visit=0.58`**，**`reject_leaf_branch_cap` 叶次 `44`**，**`visit…missing_accept=0.116`**）；**实体偏置臂**过程验收以 **`081727Z`** 为锚（**`never_visit=0.488`**，**`cap/minrel=63/10`**），**不**与纯 `probe2` 的 cap 绝对值混比。  
1. **顺序（已跑完；台账保留）**：先 **`entity_boost_alpha`**（**`0.05`→`0.1`→`0.15`→`0.3`**）；再 **`max_nodes`** 天花板（**`96` 烟测不显性增益** → **默认仍 `80`**）。**锚点**：**`α=0.05`** → **`081727Z`**；**`α=0.1` 满 500** → **`095212Z`**（**B 档**）；**`α=0.15` 满 500** → **`105756Z`**（**A 档 / 保守**）；**`α=0.3` 满 500** → **`122155Z`**（**当前默认工作点**）。  
2. **导航批验收 / 熔断**：**纯 `probe2` rule 臂**仍用旧阈：**`never_visit` 降 ≥3pp**（≤**`0.55`**）且 **`reject_leaf_branch_cap` 叶次 ≤49**。**实体偏置臂**用上节 **「阶段目标 B」**（锚 **`081727Z`**）。**上 e2e** 须：**`visit…missing_accept` 不高于 `0.12`** + 过程 **B 档及以上** + 终点 EM 见上节下限。**熔断（实体臂）**：相对 **`081727`**，`cap` 叶次 **>78** 或 **`visit…missing_accept` >0.14** → **止步**；触发后台账只记 **Δ + 判定**。  
3. **伪 visit / α 读法**：**`visited`/`never_visit` 改善**时，用 **`run_payload`/`route_decisions`** 看 **路径是否仍挤在同一浅支**（仓库**尚无**现成深度直方图字段）；若 **visit 升但分布不散**，按 **浅层匹配** 归档、**回调 α** 或改 **回溯/探索**。**叙事**：visit 动 EM 不动 → **路由未穿透生成端**；EM 动 visit 不动 → **读/生成偶然，不抬主结论**（**MI-004/005**）。  
4. **单变量 + 同口径**：跑批前 **`diff` 自检** 仅目标键一处变更；**种子 / sampler / prompt / 数据切片** 任一变 → **须重跑同协议 nav-500 基线** 再写 **Δ**。  
5. **主表与后置**：新 e2e 默认跑完 **覆盖 P0 主表 A 行 `batch_id`**，旧 **`154358Z`** 可作 **A′（历史 `probe1`）**。**金叶可达性**已由 **`122155Z`（`never_visit≈0.38`）** 等批跨过旧 **`<45%` 叙事阈值**；**实体偏置导航扫参** 已 **收口**（见上 **「P0-B′ 导航扫参收口」**）；**Router / learned 深调** 仍建议 **在 e2e 闭环评估之后** 再开；**accept 盲扫不解冻**。

**归因（常用）**：`python scripts/diagnostics/summarize_audit_failure_buckets.py outputs/reports/accept_gate_audit_<batch>.json`（**`never_visit` vs `visit_miss`** 粗分桶 + `visit_miss` 叶级 disposition）。

```bash
conda activate mamba2
cd ~/autodl-tmp/mamba2.1
git pull origin main
GEN=/root/autodl-tmp/models/Qwen2.5-7B-Instruct

# 烟测可先加： --max-samples 10（写在 run_end_to_end_batch.py 参数行上）
for cfg in \
  configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav_probe_budget2.example.json \
  configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_learned_root_blend05_probe_budget2.example.json
do
  id=$(python scripts/run_eval/run_end_to_end_batch.py \
    --config "$cfg" \
    --generator-hf-model-name "$GEN" 2>&1 \
    | sed -n 's/^__SSGS_BATCH_ID__=//p' | tail -n1)
  echo "batch_id=$id"
  python scripts/diagnostics/analyze_evidence_saturation.py \
    --registry-jsonl outputs/reports/run_registry.jsonl \
    --batch-id "$id" \
    --with-context-gold-metrics \
    --out-json "outputs/reports/evidence_saturation_${id}_ctxgold.json"
  python scripts/diagnostics/audit_accept_gate.py \
    --registry-jsonl outputs/reports/run_registry.jsonl \
    --batch-id "$id" --root .
done
```

**复现已加模版（`probe1` 满 500 + 诊断）**：

```bash
conda activate mamba2
cd ~/autodl-tmp/mamba2.1
git pull origin main

for cfg in \
  configs/experiment/navigation_batch_real_corpus_p0_probe_budget1_rule.example.json \
  configs/experiment/navigation_batch_real_corpus_p0_probe_budget1_learned_root_blend05.example.json
do
  id=$(python scripts/run_nav/run_navigation_batch.py --config "$cfg" 2>&1 \
    | sed -n 's/^__SSGS_BATCH_ID__=//p' | tail -n1)
  echo "batch_id=$id"
  python scripts/diagnostics/analyze_evidence_saturation.py \
    --registry-jsonl outputs/reports/run_registry.jsonl \
    --batch-id "$id" \
    --out-json "outputs/reports/evidence_saturation_${id}.json"
  python scripts/diagnostics/audit_accept_gate.py \
    --registry-jsonl outputs/reports/run_registry.jsonl \
    --batch-id "$id" --root .
done
```

**导航侧下一阶段（压缩）**：**P0-A′** 已闭；**P0-B′** 默认 **`122155Z`**、保守 **`105756Z`**、**单变量台账** 与 **烟测 / 满 500 分工** 见 **§6.0、§6.7** 及上文 **「P0-B′ 导航扫参收口」**。**优先**：**e2e**（同旋钮、过程优先）；**禁** `leaf_indices_required` 作弊。**未决臂**（`learned` / `cosine_probe` / `max_evidence=14`）→ **§6.7 表 ④**。

**P2（不默认）**：root 训练增强、`α>0.5` 烟测 — 见 §6.5 末、**MI-008**。

---

## 7. Learned Head 记录

当前 learned classifier 的实验意义：

1. 证明框架已具备 learned routing arm 的接入能力
2. 证明真实语料、训练数据导出、checkpoint 训练、batch 运行链均已打通
3. 当前结果显示：
- 回溯次数显著偏高
- evidence budget 基本打满
- 暂不适合作为当前主结果组

因此，当前建议把 **全树单一线性 `learned_classifier`** 定位为：

- `补充实验`
- `负结果`（若不做结构修复）
- `后续优化接口`

**补充（2026-04）**：**`learned_root_classifier` + `learned_root_blend_alpha`（与 rule 分数混合）** 在 `500` 上已恢复与启发式同量级的金叶可达性与可接受的 EM（见 **§6.5**）。主表若引用学习式 root，应写明 **混合系数 `α`** 与 **训练导出 cap**，避免与「纯 learned」混淆。

---

## 8. 当前阶段结论（压缩）

框架与真实子集已跑通；**`370M`** 进主线；轻量 routing 主对比 **`rule` / `cosine_probe`**。阶段重心已可迁到 **端到端 + 过程诊断**（详见 §9 表与 **SSGS**）。

---

## 9. 第二阶段推进方案（历史归档）

以下 **§9.1～§9.12、§10** 保留 **`batch_id` 与主表**，便于对照 Git 旧版全文；**判停阈值、MI 细则、与当前默认键** 以 **`Major_Issues_And_Resolutions_CN.md`**、**`SSGS_Research_Framework_CN.md`** 为准。**新执行项** 以 **§6.6** 为准。

当前建议正式进入第二阶段，但优先级应明确固定为：

1. 先做固定生成器的端到端评测
2. 再决定 learned head 是否值得继续主推
3. 最后再讨论 `1.4B` 与更复杂扩展

### 9.1 为什么先做端到端

原因是当前第一阶段已经回答了“导航框架能不能稳定跑通”，但还没有回答：

- 导航差异是否真的会传导到最终答案质量

如果这个问题还没有验证，直接继续做 learned head，风险是会把大量时间投入到一个尚未证明下游价值的中间模块。

### 9.2 第二阶段第一轮实验臂

建议先固定以下 arm：

1. `oracle_item_leaves`
2. `flat_leaf_concat`
3. `t1_visited_leaves_ordered`
4. `370M + rule`
5. `370M + cosine_probe`

当前不建议把 `370M + learned_classifier` 放进第二阶段第一轮主表。  
更稳妥的做法是先作为补充实验保留，等端到端结果证明“继续优化 routing 值得做”之后，再决定是否升格。

### 9.3 第二阶段第一轮指标

建议至少记录三类指标：

1. 最终答案指标
- `exact_match`
- `answer_f1` 或等价 QA 指标
- `rouge_l_f1`

2. 导航过程指标
- `nav_success_rate`
- `avg_nav_wall_time_ms`
- `avg_rollback_count`
- `avg_evidence_count`

3. 失败归因指标
- `navigation failure`
- `context construction failure`
- `generation failure`

其中需要继续坚持当前阶段已经形成的口径：

- 若 `avg_evidence_count` 继续打满 budget，则 evidence 数量仍不能作为主结论

### 9.4 第二阶段第一轮样本规模

建议先用：

- `100` 到 `200` 条真实子集

目的不是立即追求更大规模，而是先建立：

- `navigation difference -> answer difference`

这条证据链。

### 9.4.1 服务器端到端第一轮建议配置

如果服务器端准备直接使用 `7B` 左右生成模型，当前建议第一轮先固定下面四个 arm：

1. `370M + rule + t1_visited_leaves_ordered`
2. `370M + cosine_probe + t1_visited_leaves_ordered`
3. `oracle_item_leaves`
4. `flat_leaf_concat`

对应配置模板已补入：

- `configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_rule.example.json`
- `configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_cosine_probe.example.json`
- `configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_oracle_item_leaves.example.json`
- `configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_flat_leaf_concat.example.json`

当前实现中，真正可用的 `context_source` 已明确为：

- `t1_visited_leaves_ordered`
- `oracle_item_leaves`
- `flat_leaf_concat`

因此，服务器第一轮端到端对比实验已经可以按文档直接执行，不再只是停留在概念层。

但这里仍需严格区分：

- 代码与配置层面的端到端能力已经实现
- 服务器侧 `7B` 生成模型的正式对比结果尚未产出

也就是说，当前项目状态已经进入“端到端实验可执行”，但还没有进入“端到端主结果已完成”。

### 9.5 第二阶段结束后的决策规则

第一轮端到端评测结束后，建议按下面逻辑判断：

1. 如果 `oracle_item_leaves` 明显高于导航臂
- 说明当前主要瓶颈仍在导航/证据发现
- 这时继续做 learned head 更合理

2. 如果 `rule` 与 `cosine_probe` 的最终答案差异很小
- 说明当前 routing 差异尚未显著传导到下游
- 优先检查 context build、budget 和任务难度

3. 如果 `oracle_item_leaves` 与导航臂差距也不大
- 说明生成器 readout 或 prompt 可能更接近当前瓶颈
- 这时再考虑是否适度引入生成侧优化

### 9.6 `small50` 2x2 消融记录（postprocess vs anti-collapse）

为避免把“输出格式修正收益”和“导航证据质量收益”混在一起，本轮在同一数据子集上执行了 2x2 设计：

1. A: `rule + baseline`  
2. B: `rule + postprocess(constrained)`  
3. C: `rule + anti_collapse`  
4. D: `rule + anti_collapse + postprocess(constrained)`

对应 batch：

- `end_to_end_real_corpus_370m_qwen7b_rule_small50_ablation_A_20260415_230717`
- `end_to_end_real_corpus_370m_qwen7b_rule_small50_ablation_B_20260415_230947`
- `end_to_end_real_corpus_370m_qwen7b_rule_small50_ablation_C_20260415_231217`
- `end_to_end_real_corpus_370m_qwen7b_rule_small50_ablation_D_20260415_231449`

关键结果（`sample_count=50`）：

- A: `EM=0.28`, `F1=0.2903`
- B: `EM=0.28`, `F1=0.2903`
- C: `EM=0.32`, `F1=0.3370`
- D: `EM=0.32`, `F1=0.3370`

结论：

1. `postprocess` 在该设置下收益为 `0`
- A 与 B 完全一致，C 与 D 也完全一致

2. 本轮提升主要来自 `anti_collapse`
- A -> C：`EM +0.04`，`F1 +0.0467`

3. 当前阶段主瓶颈仍是导航侧证据组织，而不是输出后处理
- 后续主线应优先保留 `anti_collapse`，并将 `postprocess` 作为可审计开关保留但不主推

下一步最小行动：

1. 固定 `rule + anti_collapse` 为下一轮 `rule` 主线配置  
2. 在同一抽样上补跑 `rule + anti_collapse` vs `oracle`，重新计算 gap  
3. 若 gap 仍大，优先继续导航侧策略优化；暂不扩大生成后处理复杂度

### 9.7 `rule + anti_collapse` vs `oracle` gap 重新评估

对应 batch：

- `end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_small50_next_20260415_232753`
- `end_to_end_real_corpus_370m_qwen7b_oracle_small50_next_20260415_233018`

| 配置 | EM | F1 | gap vs oracle |
|---|---|---|---|
| rule baseline (A) | 0.28 | 0.2903 | 0.39 |
| rule + anti_collapse | 0.32 | 0.3370 | 0.343 |
| oracle | 0.68 | 0.68 | — |

结论：

1. `anti_collapse` 使 rule-oracle gap 从 `0.39` 收窄至 `0.34`（收窄约 12%）
2. gap 绝对值仍大（0.34），导航侧仍是主瓶颈
3. oracle 天花板 0.68 说明即使完美证据，7B 生成器仍有约 32% 答不对
4. 生成侧上限和导航侧上限是两个独立瓶颈，需分别优化

### 9.8 500 样本级完整对照（修复后 metrics + prompt）

对应 batch：

- `end_to_end_real_corpus_370m_qwen7b_cosine_anticollapse_small50_20260416_000238`（cosine+anticollapse, 500）
- `end_to_end_real_corpus_370m_qwen7b_rule_anticollapse_500_20260416_004405`（rule+anticollapse, 500）
- `end_to_end_real_corpus_370m_qwen7b_oracle_500_20260416_085012`（oracle, 500）

| 配置 | 样本数 | EM | F1 | 占 oracle 比例 |
|---|---|---|---|---|
| cosine + anti_collapse | 500 | 0.178 | 0.196 | 29.8% |
| rule + anti_collapse | 500 | 0.192 | 0.210 | 31.9% |
| oracle | 500 | 0.64 | 0.658 | 100% |

**与 §9.12 交叉引用**：本节表为 **`rule + anti_collapse` / `cosine + anti_collapse`** 与 Oracle（`batch_id` 见本节列表）；当时 **「本表内 rule vs cosine 差距很小」**（F1 差 **0.014**）**不可外推**为「任意协议下 cosine 都接近 overlap rule」。固定 **`context_select`**、**`overlap_k4` 与 `cosine_probe` 的 routing 三连**及 Oracle 同量级复现见 **§9.12**（该处 **cosine 较 `overlap_k4` 为 EM −0.044**）。两节 Oracle 终点均为 **`EM≈0.64`、`F1≈0.658`**，可互证上界。

小样本→大样本衰减：

| 配置 | small50 F1 | 500 F1 | 衰减幅度 |
|---|---|---|---|
| rule + anti_collapse | 0.337 | 0.210 | -38% |
| oracle | 0.68 | 0.658 | -3% |

核心结论：

1. rule-oracle gap = 0.448，导航只发挥了 oracle 潜力的 32%，改进空间极大
2. **仅限本表协议**（双臂均 **`+ anti_collapse`**）：rule vs cosine 差距很小（F1 差 **0.014**），不宜概括为「routing 永远不敏感」——**§9.12** 在 **`context_select` 对齐、`cosine_probe` vs `overlap_k4`** 下给出 **cosine 明显更差**的反例。
3. 小样本严重偏乐观（rule 衰减 38%，oracle 仅衰减 3%），后续结论必须以 500 样本为准
4. oracle 天花板 0.658 说明即使完美证据，7B 生成器仍有约 34% 答不对

### 9.9 500 样本：`rule + anti_collapse` + 实体 boost（`entity_boost_alpha`）

同一真实语料子集、`370M + Qwen7B`、端到端配置与 9.8 对齐；Oracle 批作为上界对照。

| 配置 | 样本数 | EM | F1 / ROUGE-L | `nav_success` | 相对 Oracle（EM gap） |
|---|---|---|---|---|---|
| Oracle `oracle_item_leaves` | 500 | 0.64 | ≈0.658 | — | — |
| `rule + anti_collapse`，`alpha=0.3` | 500 | 0.188 | ≈0.207 | 1.0 | ≈0.452 |
| `rule + anti_collapse`，`alpha=0.5` | 500 | 0.188 | ≈0.207 | 1.0 | ≈0.452 |

说明：

1. 与 9.8 一致：**导航完成率满**，但端到端 EM 仍远低于 Oracle，主瓶颈仍在「是否取到正确证据」而非「是否跑完导航」。
2. **`alpha=0.3` 与 `0.5` 指标完全一致**：需在 trace 上核对实体 boost 是否实际改变过排序（例如 `route_decisions` 里 `raw_router_score` vs 最终 `score`、`entity_hit_rate`）；若从未触达决策边界，则继续扫 `alpha` 收益有限。
3. 单独 5 条级的 `smoke` 批（如 `smoke5_entityalpha_*`）EM 全零属于**烟测口径/子集**，不与上表 500 条主结论混写。

### 9.10 Readout 优先诊断与判停（2026-04）

本节所涉「500 样本定量摘要、recall–readout trade-off、worst 样本中 `n_context` 扩张与 F1 塌陷、整批 `generation_error` 无效批次隔离、以及判停 / 冻结策略的完整叙述」**仅维护于** [`docs/Major_Issues_And_Resolutions_CN.md`](../Major_Issues_And_Resolutions_CN.md)（**MI-004、MI-005、MI-001**）。实验记录表与 batch 级指标仍保留于上文各节；此处不重复展开，避免双处维护。

### 9.11 500 样本 A/B：`context_select_mode`（2026-04-16）

**A）`localqwen` 主表**（`sample_count=500`，两组均 `generation_error=0/500`）：

| 配置 | `batch_id` | EM | F1 / ROUGE-L | `nav_success` |
|---|---|---:|---:|---:|
| `context_select_mode=off` | `ab500_ctxsel_off_localqwen_v2_20260416_101040Z` | `0.170` | `0.1916` | `1.0` |
| `context_select_mode=question_overlap_topk`, `context_select_k=3` | `ab500_ctxsel_overlap3_localqwen_20260416_103201Z` | `0.194` | `0.2116` | `1.0` |

**B）服务器 `370m + rule` + 本机 `Qwen2.5-7B` 快照**（`end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_rule_ctxsel_*`，`sample_count=500`，`generation_error=0`）：

| 配置 | `batch_id` | EM | F1 / ROUGE-L | `nav_success` |
|---|---|---:|---:|---:|
| `first_k`, `context_select_k=3` | `end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_first_k3_20260416_115156Z` | `0.176` | `0.1910` | `1.0` |
| `dedupe_entity_then_k`, `context_select_k=3` | `end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_dedupe_k3_20260416_121223Z` | `0.192` | `0.2030` | `1.0` |
| `question_overlap_topk`, `context_select_k=3` | `end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k3_20260416_124233Z` | `0.208` | `0.2272` | `1.0` |
| `question_overlap_topk`, `context_select_k=4` | `end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k4_20260416_140204Z` | `0.212` | `0.2337` | `1.0` |

**说明**：**A 与 B 生成端与环境不同，不得跨块比较绝对 EM 高低**；仅在各自块内读结论。**B 内**：`overlap_topk(k=3)` 已优于 `dedupe` / `first_k`（见上行）；同链路上 **`k=4` 相对 `k=3` 为 `EM +0.004`、`F1 +0.0065`**（500 主表）。与 A 中「overlap 优于 off」方向一致，但 A/B 数值不可直接对齐。

**C）服务器 B 环境、`overlap_topk` 的 `k` 扫描（`n=200` 固定子集，取 `real_corpus_navigation_batch` 前 200 条）**：仅用于筛 `k`，**不得与 500 主表数值直接对齐**；输出见 `outputs/reports/` 下各批摘要。

| `context_select_k` | `batch_id` | EM | F1 / ROUGE-L | `nav_success` |
|---:|---|---:|---:|---:|
| `2` | `pilot200_370m_rule_overlap_k2_20260416_131527Z` | `0.215` | `0.2427` | `1.0` |
| `4` | `pilot200_370m_rule_overlap_k4_20260416_132346Z` | `0.250` | `0.2796` | `1.0` |
| `5` | `pilot200_370m_rule_overlap_k5_20260416_133241Z` | `0.220` | `0.2466` | `1.0` |
| `6` | `pilot200_370m_rule_overlap_k6_20260416_134138Z` | `0.235` | `0.2598` | `1.0` |
| `8` | `pilot200_370m_rule_overlap_k8_20260416_135043Z` | `0.240` | `0.2656` | `1.0` |

**C 内粗结论**：`k=4` 在 pilot 上 EM/F1 最高；`k=8` 次之。**全量 500** 上 **`overlap_k4` 已跑完**（见 **表 B** 末行），与 **`k=3`** 对照略优。**仓库 gate**：CI 侧已增加 `demo_navigation_batch` + mock 的 **`end_to_end_batch_demo_smoke_ctxsel_overlap_k{3,4}.json`** 烟测（`tests/test_demo_ctxsel_k_smoke_batch.py`），保证 `k=3`/`k=4` 路径无 `generation_error`；**表 A 仍为历史 `k=3` 跑数**，与模版默认 `k=4` 并存时以 `batch_id`/配置字段为准。

**A 内结论**：`overlap_topk(k=3)` 相对 `off` 为 `EM +0.024`、`F1 +0.0201`。问题归因、判停口径与策略解释仅见专档 [`docs/Major_Issues_And_Resolutions_CN.md`](../Major_Issues_And_Resolutions_CN.md)（MI-004/005/006）。

**诊断摘要**：  
- **表 A、`off` vs `overlap3`（500）**（`analyze_evidence_saturation.py` + `--with-context-gold-metrics`）：导航侧两批一致（`frac_evidence_budget_saturated=1.0`、`frac_gold_leaf_ever_visited_deduped≈0.358` 等），符合「仅后处理 context、不改 trace」。生成器 context：`off` 为 `mean_n_generator_context_items≈6.89`、`mean_frac_gold_leaf_texts_in_generator_context≈0.151`；`overlap3` 为 `≈3.0`、`≈0.119`，`frac_samples_all_gold_texts_in_generator_context` 由 `≈0.012` 降至 `≈0.004`，与终点 EM 仍升并存，属 readout / 噪声与「ctx-gold 均值」非单调关系（见 MI-004）。  
- **表 B、`overlap_k3`（500）**（`batch_id=end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k3_20260416_124233Z`）：`mean_n_generator_context_items=3.0`，`mean_frac_gold_leaf_texts_in_generator_context≈0.1162`，`frac_samples_all_gold_texts_in_generator_context=0.002`；与 A 中「overlap 后 ctx-gold 均值不高」同向。完整 JSON/CSV：`outputs/reports/evidence_saturation_B_overlap_k3.json`、`.csv`。  
- **表 B、`overlap_k4`（500）**（`batch_id=end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k4_20260416_140204Z`）：导航侧与 **`overlap_k3` 同批一致**（`frac_evidence_budget_saturated=1.0`、`frac_gold_leaf_ever_visited_deduped≈0.358`、`sample_count_gold_missing_from_evidence=322` 等）。生成器 context：`mean_n_generator_context_items=4.0`，`mean_frac_gold_leaf_texts_in_generator_context≈0.1274`，`frac_samples_all_gold_texts_in_generator_context=0.002`；相对 **`k=3`**：`n_ctx` 与 ctx-gold 均值略升，与 **表 B** 终点 **`k=4` 优于 `k=3`** 同向。报告：`outputs/reports/evidence_saturation_end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k4_20260416_140204Z.json`、同名 `.csv`。

**仓库默认（2026-04-16，2026-04-17，2026-04-18）**：`configs/experiment/` 下凡 `context_source` 为 `t1_visited_leaves_ordered` 或 `flat_leaf_concat`、且为 **`question_overlap_topk`** 的模版，**`context_select_k` 已 bump 为 `4`**（与 **表 B** 全量 500 结论一致）；**`first_k3` / `dedupe_k3` 例题**仍为 `context_select_k=3`（与臂名一致）。**`oracle_item_leaves` 臂**显式 `context_select_mode=off`。复现 **表 A** 请仍使用 `k=3` 的历史 `batch_id` 或自建配置，勿与默认模版混读。

### 9.12 B 链：routing 对照（500，本机 Qwen，`context_select` 已对齐）

**与 §9.8**：两节均为 **500 × `370m` + 本机 Qwen**、Oracle **`EM≈0.64`、`F1≈0.658`**（本节 Oracle `batch_id` 为 **`…oracle_20260416_155420Z`**，与 **§9.8** 表内 Oracle **`…oracle_500_20260416_085012`** 不同跑次、指标同量级）。**§9.8** 比较的是 **`rule + anti_collapse` / `cosine + anti_collapse`**；本节比较的是 **固定 `context_select` 后** 的 **`overlap_k4`（rule 路由）** vs **`cosine_probe`** vs Oracle，**不得与 §9.8 的绝对 EM 混读**。

同一 **`sample_count=500`**、**`370m` + 本机 `Qwen2.5-7B`**、**`nav_success=1.0`**；**rule / cosine** 为 **`t1_visited_leaves_ordered` + `question_overlap_topk` + `k=4`**，**Oracle** 为 **`oracle_item_leaves` + `context_select off`**（与 `end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_*.example.json` 一致）。

**与 §6.7「`visit_a030` / `entity_match` 栈」导航批的 `exact_match_rate` 不得混读**：本节表内 **EM** 来自 **端到端**：**`run_generator=True`**，`trace` 按 **`eval_mode=generation`** 用 **生成答案** 与 **参考答案** 打分（见 **`run_navigation_sample`**）。**§6.7** 中 **`run_navigation_batch`、配置未开生成器** 时，默认 **`eval_mode=retrieval`**：用 **`context_select` 之后的第一条上下文文本**（若无则回退首条 **accepted evidence**）与 **参考答案** 做 **exact_match / F1 / ROUGE-L**——这是仓库内约定的 **粗「检索式」proxy**，**数值尺度与含义均不同于本节 7B 生成 EM**。此外 **路由与读侧也不同**：本节 **`rule` = `overlap_k4`（`question_overlap_topk`）**；**`122155Z` / `081150Z`** 为 **`visit_rule` + `entity_boost_alpha` + `question_entity_match_topk`** 等 **P0-B′ 协议**。因此 **§9.12 中 `cosine_probe` 弱于 `overlap_k4`** 与 **§6.7 中 `081150Z` 优于 `122155Z`（在检索 proxy 上）** 可以同时成立；写作时应用 **小节名 + `batch_id` + `eval_mode` + 路由键** 四元组标明口径。

| 臂 | `batch_id` | EM | F1 / ROUGE-L | `nav_success` | EM 占 Oracle |
|---|---|---:|---:|---:|---:|
| `rule`（`overlap_k4` 主表） | `end_to_end_real_corpus_370m_qwen7b_rule_ctxsel_overlap_k4_20260416_140204Z` | `0.212` | `0.2337` | `1.0` | **≈33.1%** |
| `cosine_probe` | `end_to_end_real_corpus_370m_qwen7b_cosine_probe_20260416_153243Z` | `0.168` | `0.1847` | `1.0` | **≈26.3%** |
| `oracle_item_leaves` | `end_to_end_real_corpus_370m_qwen7b_oracle_20260416_155420Z` | **`0.64`** | **`≈0.658`** | `1.0` | **100%** |

**与 `rule` 对照（cosine − rule）**：`EM -0.044`（相对约 **−20.8%**），`F1 -0.0490`（相对约 **−21.0%**）。

**相对 Oracle 的 F1 gap（便于与 §9.8「rule-oracle gap」对照）**：**Oracle − `rule`** ≈ **`0.658 − 0.2337 ≈ 0.424`**；**Oracle − `cosine_probe`** ≈ **`0.658 − 0.1847 ≈ 0.473`**。EM gap 同向：**`0.64 − 0.212 = 0.428`**、**`0.64 − 0.168 = 0.472`**。

**Oracle 补齐后的结论（本批 B 链三连）**：**Oracle EM = `0.64`**，相对 **`rule` EM = `0.212`** 的差 **≈`0.428`**，远高于事先约定的分叉阈值 **`0.03`**，故 **「金证据上界」显著高于当前 overlap 路由实际送入生成器的上下文**——与 **9.11** 一致，主矛盾仍在 **证据发现 / 路由与接受**，而非 **`nav_success`（三臂均为 `1.0`）**。在此前提下，**`cosine_probe` 仍低于 `rule`**，且占 Oracle 比例由 **≈33.1%** 跌至 **≈26.3%**，优先叙事为：**`cosine_probe` 相对 `overlap_k4` 进一步伤害证据路由**；同时 **Oracle F1 ≈ `0.658`** 表明即使用金叶上下文，**7B 生成仍有约 34% 答不对**（与 **9.8** 全量 Oracle 叙述同量级）。本批 **`avg_nav_wall_time_ms`**：Oracle **`≈1128`**，`cosine` **`≈1178`**，`rule` 批见对应 `batch_summary`（量级相近，不构成主差异）。

**过程指标（`analyze_evidence_saturation.py --with-context-gold-metrics`，与上表同一 `batch_id`）**：落盘 **`outputs/reports/evidence_saturation_9p12_rule_overlap_k4_ctxgold.json`**、**`…_cosine_probe_ctxgold.json`**（及同名 **`_per_sample.csv`**）。摘要对比如下（**`diag_*` 均为 500/0/500**，树与 ctx-gold 可解析）。

| 指标 | `rule`（`overlap_k4`） | `cosine_probe` |
|------|------------------------:|---------------:|
| `frac_evidence_budget_saturated` | `1.0` | `1.0` |
| `mean_n_evidence` / `mean_max_evidence` | `8.0` / `8.0` | `8.0` / `8.0` |
| `frac_gold_leaf_ever_visited_deduped` | **`0.358`** | **`0.246`** |
| `frac_gold_in_accepted_evidence` | **`0.356`** | **`0.244`** |
| `sample_count_gold_missing_from_evidence` | **`322`** | **`378`** |
| `frac_saturated_among_gold_missing` | `1.0` | `1.0` |
| `mean_unique_entities_in_evidence` | `4.492` | `5.84` |
| `mean_evidence_same_entity_as_first` | `2.53` | `1.552` |
| `mean_n_generator_context_items` | `4.0` | `4.0` |
| `mean_frac_gold_leaf_texts_in_generator_context` | **`≈0.1274`** | **`≈0.0899`** |
| `frac_samples_all_gold_texts_in_generator_context` | `0.002` | `0.008` |

**解读（与终点 EM 对齐）**：**`cosine_probe`** 相对 **`rule`**，**金叶子曾访问比例**、**金进 accepted evidence**、**生成器 context 含金文均值**均**更低**，**`gold_missing_from_evidence` 多 56 条**——与 **EM `0.168` vs `0.212`** 同向，支持 **「cosine 路由在证据发现与送入生成器的链路上弱于 overlap」**，而非仅 `context_select` 后处理差异。**`rule`** 侧导航与生成器摘要与 **9.11 表 B `overlap_k4`** 一致（如 **`frac_gold_leaf_ever_visited_deduped=0.358`**、**`gold_missing=322`**），可互证同批。**`frac_samples_all_gold_texts_in_generator_context`** 在 cosine 上略高（`0.008` vs `0.002`）但绝对仍极低，**不足以抵消** **`mean_frac_gold_leaf_texts_in_generator_context`** 的劣势。

---

## 10. 当前建议与下一步（2026-04 更新）

若目标是「在固定 500 条主表上把结论写稳、并定位下一刀改哪里」，建议按优先级：

### 10.1 证据饱和与金叶子可达性（优先）

单纯把 `max_evidence` / `context_max_items` 提到 `16` 仍无法解决时，应默认怀疑 **同质证据顶满预算** 或 **探索路径从未靠近金叶子**，而不是「槽位不够」。

仓库内离线汇总脚本为 `scripts/diagnostics/analyze_evidence_saturation.py`（按 `run_registry.jsonl` 的 `batch_id` 聚合每条 `outputs/runs/<run_id>/run_payload.json`）。需要区分「导航侧证据」与「生成器实际 context」时加 `--with-context-gold-metrics`；输出路径用 `--out-json` / `--per-sample-csv`。**可执行命令不在本文件维护**（避免路径与 `batch_id` 随环境变化导致文档过时）。

关注摘要中的 **`frac_evidence_budget_saturated`**、**`mean_unique_entities_in_evidence`**，以及在 batch 传入 `positive_leaf_indices` 时的 **`frac_gold_leaf_ever_visited_deduped`** 与 **`frac_gold_in_accepted_evidence`**，再决定是加强 **anti-collapse / 多样性** 还是动 **接受阈值与探索顺序**。

#### 10.1.1 决策检查清单（锚定 **B、`overlap_k4`（500）** 已跑诊断）

以下数字来自 **9.11 诊断摘要** 中 **`overlap_k4`** 与导航侧一致项，用作「下一刀」优先级，而非再扫 `context_select`：

1. **`frac_evidence_budget_saturated = 1.0` 且 `mean_n_evidence = mean_max_evidence = 8`**：优先审视 **证据同质 / anti-collapse** 与 **接受后是否仍挤满同质叶子**，而不是单纯加 `max_evidence`。  
2. **`sample_count_gold_missing_from_evidence = 322`（500 中）且 `frac_saturated_among_gold_missing = 1.0`**：主矛盾在 **金叶子未进 accepted evidence**；优先动 **探索顺序 / 接受阈值 / router 与实体多样性**，使金叶子更可能进入 evidence。  
3. **`mean_evidence_same_entity_as_first` 较高**（与 **`mean_unique_entities_in_evidence`** 对照）：若「同实体重复」突出，与 **anti-collapse / 去重接受** 联动设计实验臂。  
4. **`frac_gold_leaf_ever_visited_deduped` 与 `frac_gold_in_accepted_evidence` 仍偏低**：若 visited 尚可但 accepted 差，动 **接受**；若 visited 即差，动 **导航探索与 routing**。  
5. 每轮改参仍须同时报 **终点 + 过程**；若出现 **过程升、终点降**，按专档 **MI-004 / MI-005** 判停。

### 10.2 与第二阶段叙事对齐

1. **主表**：继续以 **500 条** `rule + anti_collapse` / `cosine + anti_collapse` vs **Oracle** 为锚；小样本仅作消融提示，不写主结论。
2. **实体 boost**：在 10.1 有 trace 证据前，不把 `alpha` 超参扫作为主工作量。
3. **生成与评测口径**：端到端配置中已支持 `eval_mode`、`report_dir` 等字段；新跑批次应在 `run_registry.jsonl` 中可追溯，便于与诊断脚本联动。
4. **`context_select` 消融队列（已闭合）**：服务器 **B** 上三模式 + **`overlap_k3`/`k4`（500）** 与 **pilot200**、**`overlap_k4` 诊断** 已齐（**9.11**）。**`k=8` 全量 500** 仅作备选（ROI 低）。
5. **P0-A（证据 / 导航）**：按 **10.1.1** 从 **`gold_missing` + 饱和 + 实体重复`** 定下一组可审计实验臂（anti-collapse / 接受 / 探索），冻结 **`mrs/pem` 盲扫**。  
6. **P0-B（第二阶段三连跑，B 链 500）**：在固定 **`context_select`**（**rule / cosine** 为 **`overlap_topk` + `k=4`**，**Oracle** 为 **`off`**）下，用仓库脚本依次跑 **rule → `cosine_probe` → `oracle_item_leaves`**，注入本机 Qwen 目录：  
   - `python scripts/run_eval/run_b_chain_phase2_three_arm.py --generator-hf-model-name '<本机Qwen目录>'`  
   - 可选先 `--dry-run` 检查生成的 `outputs/reports/tmp_phase2_configs/phase2_patch_*.json`。  
   - 跑完将三条 **`batch_id`** 与 EM/F1 写入 **9.12**（**2026-04-16** 三连：`rule` / `cosine_probe` / `oracle_item_leaves` 已齐，见表）。  
   - **生成端须指本机 Qwen**（**MI-001**）；`git pull` 不通时用 **MI-002** 单文件 / ZIP 同步脚本与配置。  
   - 若批跑到一半报 **`OSError: 28`（磁盘满）**，见专档 **MI-007**；清空间后从失败臂重跑（串联脚本会从 **rule** 再跑一遍，可接受重复或改用手动两条配置只跑 **cosine / oracle**）。
7. **三连已闭合后的下一刀**：**P0-B** 与 **§9.12** 终点表已齐；**§9.12** 已补 **`rule` / `cosine_probe` 双批 `--with-context-gold-metrics`** 过程表与落盘路径。叙事与判停口径见 [`SSGS_Research_Framework_CN.md`](SSGS_Research_Framework_CN.md) **§11.0.5**。**下一执行项（P0-A）**：按 **§10.1.1** 与下节 **§10.2.1** 设计并跑可审计臂（仍冻结 **`mrs/pem` 盲扫**），每臂仍须 **终点 + 本脚本过程指标** 同报。

#### 10.2.1 P0-A 候选臂（外部草案的批判修订，与指标契约对齐）

`scripts/diagnostics/analyze_evidence_saturation.py` 中：**`n_evidence` / `saturated` / `frac_gold_in_accepted_evidence` / `sample_count_gold_missing_from_evidence`** 来自 **`trace.evidence_texts`** 与 **`event_log`（`accept_evidence`）**，由 **`SSGSController` 导航闭环**写入；**`mean_frac_gold_leaf_texts_in_generator_context`** 等则来自 **写入生成器前的 `context_texts` / `generator_evidence_texts`**，受 **`phase_a_runner._select_context_items`（`context_select_*`）** 影响（见 **MI-003**、脚本 `analyze_payload`）。因此：

- **仅改 `context_select_mode` / visited→context 管线**：可抬 **生成器 ctx-gold / EM**，**一般不能**抬 **`frac_gold_in_accepted_evidence` 或压低 `gold_missing`（证据侧）**——除非同时改 Controller 或 trace 定义。外部草案把 A1/A2 的「预期 ↑ `frac_gold_in_accepted_evidence`」与 **`gold_missing≈378`** 绑在一起 **不成立**（**378** 为 **§9.12 `cosine_probe`** 批；**`rule`** 为 **322**）。
- **真要动「接受」**（证据槽里的叶子集合）：需动 **Controller 接受逻辑 / `max_evidence` 与 accept 策略 / Router 带来的访问序**，**不是**「仅 `phase_a_runner` 后处理」能等价替代；排错成本确实更高，应单独命名（例如 **P0-A′**），与 **MI-006** 的「后读排序」解耦。
- **「优先接受而非探索」**在工程风险上可接受，但须把目标指标说对：**先做后读排序（低成本）→ 看 EM 与 `mean_frac_gold_leaf_texts_in_generator_context`**；若 **`gold_missing` / `frac_gold_in_accepted`** 仍差，再评估是否值得动 Controller。

| 臂（修订命名） | 核心假设 | 实际作用域 | 主要可验证指标（本脚本） | 风险与备注 |
|:---|:---|:---|:---|:---|
| **B1：`question_entity_match_topk`（已实现）** | 纯词 overlap 易受泛词干扰；按 **问题实体命中比例** 对 **visited→context** 重排，可抬高 **ctx-gold** | **`phase_a_runner._select_context_items`**，不改 Controller | **`mean_frac_gold_leaf_texts_in_generator_context`**、EM/F1；**不承诺** **`frac_gold_in_accepted_evidence`** | 低～中（单元测 `test_run_navigation_sample_context_select_question_entity_match_topk`；全量 500 用模版 **`configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_rule_ctxsel_entity_match_k4.example.json`**，与 **`…_rule.example.json`** 仅差 **`context_select_mode`** 与 **`run_id_prefix` / `batch_id_prefix`**） |
| **B2：`context_select_pool_max_items`（已实现）** | 金叶已出现在 **visited** 但落在 **`context_max_items`** 截断之后未进入打分池；**扩大构建上限**再经 overlap/entity 取 top-`k`（**非** Controller「延迟 accept」） | **`phase_a_runner`**：`run_navigation_sample` 对 `_build_context_from_trace` 传入 **`max(context_max_items, pool_max)`**；**`off`** 时忽略 pool | **`mean_frac_gold_leaf_texts_in_generator_context`**、EM；**不承诺** **`frac_gold_in_accepted_evidence`** | 中（单测 `test_context_select_pool_max_items_widen_overlap_candidates`；全量见 **`…rule_ctxsel_entity_match_k4_pool20.example.json`**） |
| **B3：句级去重 / Jaccard anti-collapse** | 同质块挤占槽位；在 **证据文本或 context 列表** 上滤近重复 | **`_apply_evidence_controls` 或读侧去重** | **`mean_evidence_same_entity_as_first`**、过程噪声；若动在 Controller 前则可能影响 **`evidence_texts`** 长度 | 中高（阈值误杀；**`mean_pairwise_evidence_jaccard` 需先加诊断或离线抽样**） |
| **B4：`rule` 分数混合 `cosine_probe`（小权重）** | 微调下钻序以抬高 **`frac_gold_leaf_ever_visited`** | **Router / Controller** | **`frac_gold_leaf_ever_visited_deduped`**、`gold_missing`（证据侧） | **高**：§9.12 已示 **纯 cosine 劣于 rule**，混合可能改善或 **进一步伤**；需 **500 + 全量过程指标**，列为 **P0-B** 更妥 |

**判停与优先级（修订）**：  
1. **先做 B1 → B2**（仍锚 **`rule` + `overlap_k4` 等价协议**）：以 **EM / `mean_frac_gold_leaf_texts_in_generator_context`** 为主判据；**`nav_success`** 维持 **≥0.98**（或主表约定 **1.0**）即停损。  
2. **`frac_evidence_budget_saturated` 显著 < 1.0**：未必等于「探索不足」，也可能是 **`max_evidence` 下调 / 提前停导航**；须结合 **`mean_n_evidence`** 与 **`nav_success`** 解读，**不宜**作唯一熔断条件。  
3. **B3** 在 B1/B2 **ctx-gold 仍顶不上去** 或 **同质化指标**明确后再开。  
4. **B4** 仅在 **读侧触顶**（例如 Oracle 占比长期 **<40%** 且 gold_missing 仍主导）再纳入，并与 **§9.12 cosine** 结论 **对打** 设计（权重网格要小）。

**可粘贴摘要（取代未校验的外部文案）**：**P0-A 优先验证读侧重排与 visited 窗口（B1/B2），以抬升生成器 ctx-gold 与 EM，且不冒认能单独解决 `gold_in_accepted` / `gold_missing`；证据槽级问题保留为 Controller 侧 P0-A′。混合 cosine（B4）因 §9.12 已证 cosine 劣于 rule，降级为高成本探索臂。**

### 10.3 批判性接收（RAPTOR / IRCoT 启发）

本轮对外部方法的接收原则：

1. **直接采纳**：`collapsed-tree` 思路对应“排序优先于访问顺序”  
- 以 `context_select_mode` 后处理做最小实现，先验证 readout 假设，不改 Controller 主逻辑。

2. **部分采纳**：IRCoT 的逐跳检索启发  
- 当前仅考虑“轻量 query hint（已访问实体）”作为后续方向；不引入完整 CoT-检索循环，避免变量爆炸。

3. **暂不采纳**：重型离线聚类摘要树重构  
- RAPTOR 式 GMM+LLM 摘要构树成本高、变量多，留到后续阶段；当前先榨干现有树结构与 readout 改进空间。

### 10.4 当前冻结策略（防盲扫）

阈值冻结、`context_select_mode` 单变量主线、终点与过程指标同报、以及「过程升终点降」时的判停细则，**仅维护于** [`docs/Major_Issues_And_Resolutions_CN.md`](../Major_Issues_And_Resolutions_CN.md)（**MI-004、MI-005、MI-006**）。

### 10.5 服务器代码更新（GitHub 不稳定时）

无 `.git` 的 ZIP 部署、大目录迁移与单文件同步等操作口径，**仅维护于** [`docs/Major_Issues_And_Resolutions_CN.md`](../Major_Issues_And_Resolutions_CN.md)（**MI-002**）。

---

## 11. 历史备忘（原「当前建议」骨架）

若当前阶段的目标是「从导航闭环转入可发表的第二阶段验证」，长期仍需要：

1. 固定第二阶段实验臂与指标口径  
2. 固定生成器与 context build 合同  
3. 在代表性子集上保持端到端复跑节奏  
4. 用端到端 + 过程诊断共同决定 learned head 等资源是否值得继续投入
