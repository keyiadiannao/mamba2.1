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

---

## 7. Learned Head 记录

当前 learned classifier 的实验意义：

1. 证明框架已具备 learned routing arm 的接入能力
2. 证明真实语料、训练数据导出、checkpoint 训练、batch 运行链均已打通
3. 当前结果显示：
- 回溯次数显著偏高
- evidence budget 基本打满
- 暂不适合作为当前主结果组

因此，当前建议把 learned classifier 定位为：

- `补充实验`
- `负结果`
- `后续优化接口`

而不是当前阶段主表方案。

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

小样本→大样本衰减：

| 配置 | small50 F1 | 500 F1 | 衰减幅度 |
|---|---|---|---|
| rule + anti_collapse | 0.337 | 0.210 | -38% |
| oracle | 0.68 | 0.658 | -3% |

核心结论：

1. rule-oracle gap = 0.448，导航只发挥了 oracle 潜力的 32%，改进空间极大
2. rule vs cosine 差距很小（F1 差 0.014），当前路由策略不是关键区分点
3. 小样本严重偏乐观（rule 衰减 38%，oracle 仅衰减 3%），后续结论必须以 500 样本为准
4. oracle 天花板 0.658 说明即使完美证据，7B 生成器仍有约 34% 答不对

---

## 10. 当前建议

如果当前阶段的目标是“从导航闭环转入可发表的第二阶段验证”，那么现在最值的工作不是继续横向扩模块，而是：

1. 固定第二阶段实验臂
2. 固定生成器和 context build 口径
3. 先跑一轮 `100-200` 条端到端评测
4. 用端到端结果决定 learned head 的真实优先级
