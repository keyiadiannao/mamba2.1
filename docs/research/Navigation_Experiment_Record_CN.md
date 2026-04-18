# Navigation 实验记录（阶段一）

## 1. 文档目的

本记录用于固定当前阶段导航实验的关键事实，避免结果只保留在终端输出和临时交流里。  
目标不是完整论文草稿，而是形成一份可追溯、可复述、可继续追加的实验台账。

---

## 2. 当前阶段实验目标

当前阶段的实验目标已经从“框架能否跑通”推进到“框架在真实公开数据子集上是否稳定运行”。

当前主线任务：

1. 打通真实数据入口
2. 跑通 `370M` 预训练导航器
3. 比较轻量 routing 方案
4. 确定哪些结果可以进入主结果表

---

## 3. 数据与预处理链

当前真实数据来源：

- `2WikiMultiHopQA`

当前采用的标准预处理链：

1. `extract_2wiki_subset.py`
- 从完整 `2Wiki` 文件中抽取可复现实验子集

2. `prepare_2wiki_subset.py`
- 把 `2Wiki` 原始格式转成 `wiki-longdoc` 中间格式

3. `prepare_wiki_longdoc_subset.py`
- 把 `wiki-longdoc` 样本展开为 `corpus jsonl + qa jsonl`

4. `build_navigation_inputs_from_jsonl.py`
- 生成 `tree payload + navigation batch manifest`

---

## 4. 当前主线配置

当前主线只保留两组：

1. `370M + rule`
2. `370M + cosine_probe`

当前不进入主结果表的组：

1. `smoke + rule`
- 仅用于验证真实链路可运行

2. `370M + learned_classifier`
- 已完成系统接入与真实语料实验
- 但当前表现更适合作为补充实验或负结果

---

## 5. 关键实现进展

当前阶段已经完成的关键工程节点：

1. `Mamba` 导航器已支持 smoke 后端与 `370M hf_pretrained`
2. batch runner 已修复为“批内共享 controller”
- 避免每个样本重复加载 `370M` 权重

3. trace / registry / navigation summary 已支持批次追踪
4. 真实语料预处理链已可在服务器完整执行

---

## 6. 关键实验记录

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
- 训练：`scripts/run_nav/train_learned_router.py`，**`--loss listwise_softmax`**（默认），写出 **`configs/router/learned_root_router_real_corpus.json`**。
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

### 6.6 下一阶段计划（维护决策，2026-04）

以下顺序为 **默认主线**；**不再把 `α>0.5` 常规化**（见 §6.5 末段）。

**P0（优先，验证「导航改进是否传到终点」）**

1. **端到端一批（`500` 或当前主 manifest）**  
   - **臂 A**：冻结启发式 **`routing_mode=rule`**（与现 frozen 模版一致）。  
   - **臂 B**：**`learned_root_classifier` + `learned_root_blend_alpha=0.5`**（与现默认一致）。  
   - **固定**：同 **`samples_path`**、同生成器配置、同 **`context_select_*`**；只改 **`routing_mode` / `learned_root_blend_alpha` / `batch_id_prefix`**。  
   - **验收**：`generation_error` 统计、`EM/F1`、可选 **`analyze_evidence_saturation.py --with-context-gold-metrics`** 看 ctx-gold。  

   **已落盘模版（仓库内，与导航 frozen 协议对齐）**：

   - 臂 A：`configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav.example.json`  
   - 臂 B：`configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_learned_root_blend05.example.json`  

   **终端（仓库根目录）**：生成器路径可用 **`--generator-hf-model-name`** 或环境变量 **`GENERATOR_HF_MODEL_NAME`** 覆盖 JSON 中的 Hub id（与 `run_end_to_end_batch.py` 帮助一致）。建议 **先烟测再全量**：

   ```bash
   cd /root/autodl-tmp/mamba2.1
   git pull origin main

   # 烟测（各 10 条，确认无 generation_error 堆积）
   python scripts/run_eval/run_end_to_end_batch.py \
     --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav.example.json \
     --max-samples 10

   python scripts/run_eval/run_end_to_end_batch.py \
     --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_learned_root_blend05.example.json \
     --max-samples 10

   # 全量 500（去掉 --max-samples）
   python scripts/run_eval/run_end_to_end_batch.py \
     --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_rule_frozen_nav.example.json

   python scripts/run_eval/run_end_to_end_batch.py \
     --config configs/experiment/end_to_end_batch_real_corpus_server_mamba_370m_qwen7b_p0_learned_root_blend05.example.json
   ```

   **登记结果**：每臂终端会打印 **`batch_id`**；摘要写入 **`outputs/reports/end_to_end_batches/<batch_id>/batch_summary.json`**，并追加 **`outputs/reports/end_to_end_batch_summary.jsonl`**。  
   **金叶 / 证据（与导航批同一脚本）**：

   ```bash
   python scripts/diagnostics/analyze_evidence_saturation.py \
     --registry-jsonl outputs/reports/run_registry.jsonl \
     --batch-id '粘贴该臂batch_id' \
     --out-json outputs/reports/evidence_sat_p0_<臂标签>.json

   # 可选：生成器 context 含金文
   python scripts/diagnostics/analyze_evidence_saturation.py \
     --registry-jsonl outputs/reports/run_registry.jsonl \
     --batch-id '粘贴该臂batch_id' \
     --with-context-gold-metrics \
     --out-json outputs/reports/evidence_sat_p0_<臂标签>_ctxgold.json
   ```

   **`generation_error` 粗扫**（按 `run_registry.jsonl` 里该 `batch_id` 的 `output_run_dir` 读 `run_payload.json`）：

   ```bash
   python -c "
   import json
   from pathlib import Path
   bid = '粘贴batch_id'
   reg = Path('outputs/reports/run_registry.jsonl')
   errs = []
   nrows = 0
   for line in reg.read_text(encoding='utf-8').splitlines():
       if not line.strip():
           continue
       r = json.loads(line)
       if r.get('batch_id') != bid:
           continue
       nrows += 1
       od = r.get('output_run_dir')
       if not od:
           continue
       p = Path(od) / 'run_payload.json'
       if not p.is_file():
           continue
       d = json.loads(p.read_text(encoding='utf-8'))
       ge = (d.get('trace') or {}).get('generation_error')
       if ge:
           errs.append((str(p), str(ge)))
   print('registry_rows_for_batch:', nrows)
   print('payloads_with_generation_error:', len(errs))
   for path, msg in errs[:20]:
       print(path, '->', msg[:200])
   "
   ```

   **排错：`exact_match_rate` / `avg_answer_f1` 全为 0 但 `nav_success_rate=1`**：常见原因是 **`reference_answer` 在 manifest 中为列表**（如 2Wiki 多答案），旧版 `run_end_to_end_batch` 用 ``str(list)`` 或管线未规范化，导致 **不参与 EM/F1 打分** 或参考串错误。仓库已统一 **`normalize_reference_for_scoring`**（`src/evaluation/reference_answer.py`）；**请 `git pull` 后重跑该端到端批** 再比 EM。若仍全零，再查 **`trace.generation_error`** 与 **`reference_answer`** 字段是否为空（任一样本 `run_payload.json`）。

   **P0 端到端 `500` 实测（`Qwen2.5-7B-Instruct`，同 manifest / 同导航协议模版；2026-04-17）**  

   | 臂 | `batch_id` | `exact_match_rate` | `avg_answer_f1` | `avg_nav_wall_time_ms` |
   |:---|:---|---:|---:|---:|
   | A `rule`（frozen nav） | `end_to_end_p0_real_corpus_370m_qwen7b_rule_frozen_nav_20260417_154358Z` | **0.186**（93/500） | **≈0.205** | **≈1247** |
   | B `learned_root` + `blend=0.5` | `end_to_end_p0_real_corpus_370m_qwen7b_learned_root_blend05_20260417_160609Z` | **0.200**（100/500） | **≈0.221** | **≈1301** |

   **结论**：**臂 B 相对臂 A：EM 与 F1 略升，`nav_ms` 略增**；与证据侧金叶摘要（`visited≈0.454` vs `≈0.428`）同向，支持 **「混合 root 不仅修导航过程指标，也传导到端到端答案质量」** 的阶段性判断（仍属小幅差距，不宜过度外推）。

2. **导航侧回归（P0-2，推荐在端到端 `500` 之后立刻做）**  
   - **目的**：用 **前 `200` 条**（或全 manifest 的 **`pilot200` 切片**，若你本地另有该文件）跑 **仅导航批**（无生成器负载），对照 **`frac_gold_leaf_ever_visited_deduped` / `gold_missing` / `nav_ms`**，防止端到端 `500` 子集偶然。  
   - **协议**：与 **§6.6 项 1** 的 P0 端到端 **frozen nav** 一致（`context_select_pool_max_items=20`、`explore_root_probe_*`、`explore_top_m_root_children=0`）；两臂仅差 **`routing_mode` / `learned_root_*`**。  
   - **仓库模版**（`run_navigation_batch.py` + **`--max-samples 200`**）：  
     - 臂 A：`configs/experiment/navigation_batch_real_corpus_p0_frozen_nav_reg200_rule.example.json`  
     - 臂 B：`configs/experiment/navigation_batch_real_corpus_p0_frozen_nav_reg200_learned_root_blend05.example.json`  

   ```bash
   cd /root/autodl-tmp/mamba2.1
   git pull origin main

   python scripts/run_nav/run_navigation_batch.py \
     --config configs/experiment/navigation_batch_real_corpus_p0_frozen_nav_reg200_rule.example.json \
     --max-samples 200

   python scripts/run_nav/run_navigation_batch.py \
     --config configs/experiment/navigation_batch_real_corpus_p0_frozen_nav_reg200_learned_root_blend05.example.json \
     --max-samples 200
   ```

   终端会打印 **`batch_id`**；摘要目录 **`outputs/reports/batches/<batch_id>/batch_summary.json`**。金叶汇总：

   ```bash
   python scripts/diagnostics/analyze_evidence_saturation.py \
     --registry-jsonl outputs/reports/run_registry.jsonl \
     --batch-id '粘贴该臂 batch_id' \
     --out-json outputs/reports/evidence_sat_nav_p0_reg200_<臂标签>.json
   ```

   **登记（`N=200` 前缀切片，`manifest_sample_count=500`；2026-04-18）**：

   | 臂 | `batch_id` | `nav_success_rate` | 金叶 `frac_gold_leaf_ever_visited_deduped` | `sample_count_gold_missing_from_evidence` | `avg_nav_wall_time_ms` |
   |:---|:---|:---:|:---:|:---:|:---:|
   | A `rule` frozen | `nav_p0_reg200_rule_frozen_20260418_014016Z` | **1.0** | **0.41** | **130** | **≈1363** |
   | B `learned_root` `α=0.5` | `nav_p0_reg200_learned_root_blend05_20260418_014536Z` | **1.0** | **0.445** | **122** | **≈1353** |

   **金叶补充（同批 `analyze_evidence_saturation.py` summary）**：`frac_gold_in_accepted_evidence` **0.35**（A）vs **0.39**（B）；`mean_gold_index_first_in_visits` **≈2.46**（A）vs **≈1.79**（B）。证据槽饱和两臂均为 **`frac_evidence_budget_saturated=1.0`**、`mean_n_evidence=8`。

   **检索口径打分**（导航批默认无生成器，`eval_mode=retrieval`）：`exact_match_rate` **0.11**（22/200）vs **0.125**（25/200）；`avg_answer_f1` **≈0.121** vs **≈0.137**。

   **结论**：与 **§6.6 项 1** 端到端 `500` **同向**——臂 B **金叶 visited / accepted 更高、`gold_missing` 更少**，检索 EM/F1 略升；本切片上 **`nav_ms` 略低于臂 A**（与全量端到端里 B 略慢不完全一致，属子集方差可接受范围）。**P0-2 回归通过**。

**P1（仅当 P0 显示瓶颈仍在「读侧 / 上下文」）**

3. **读侧与候选池**（与 MI-003 / MI-006 叙事一致）：在导航臂已固定的前提下，评估 **`context_select_mode` / `context_select_pool_max_items`** 等 **不改变 Controller 接受逻辑** 的改动。  

**P2（不默认排期）**

4. **学习式 root 再增强**：仅当产品/论文需要 **更强 learned 分量** 时，再开 **`max-root-children`↑、cap 外 hard negatives、listwise 目标细化`**；**不与 `α>0.5` 扫参混在同一里程碑**。  

5. **`α` 上界研究**：仅论文需要时 **烟测 `0.6`～`0.7` + 金叶闸门**，不纳入常规迭代。

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

## 8. 当前阶段结论

当前可以正式确认：

1. 基础导航框架已经构建完成
2. 真实公开数据子集实验已经跑通
3. `370M` 预训练导航器已经进入主线
4. 主线 routing 目前可稳定比较的对象是：
- `rule`
- `cosine_probe`

因此，当前阶段已经从“框架搭建期”进入“结果整理与写作收束期”。

---

## 9. 第二阶段推进方案

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
