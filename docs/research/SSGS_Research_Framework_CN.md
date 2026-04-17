# SSGS 树状 RAG 研究框架文档

## 1. 课题一句话定义

本研究旨在构建一个面向树状 RAG 的、基于 Mamba 状态快照回溯的导航框架；其首要目标是实现可运行、可审计、可复现的检索闭环，而不是先追求最终生成性能。第一阶段主线以树状结构为核心，后续再谨慎扩展到可展开为搜索树的更一般结构化检索空间。

---

## 2. 我们到底在研究什么

本课题研究的不是：

- 让 Mamba 直接替代 Transformer 做答案生成
- 单纯追求一个更高榜单分数的 RAG 系统
- 在没有完整系统闭环之前就直接宣称速度或精度优势

本课题真正研究的是：

在树状知识结构中，能否利用 Mamba/SSM 的固定大小隐状态及其快照恢复能力，构建一个支持低成本回溯的导航器，并与生成器彻底解耦，从而形成一个可运行、可审计、可比较的树状 RAG 框架；并在后续阶段评估这一机制是否可迁移到按查询展开的结构化图搜索空间。

换句话说，我们研究的是导航侧的状态管理与搜索控制，而不是生成侧能力本身。

---

## 3. 背景与问题动机

### 3.1 为什么不是只做平面 RAG

平面 RAG 通常基于扁平 chunk 检索，适合短上下文和单跳问答，但在以下场景存在局限：

- 长文档问答：单个 chunk 很难覆盖全文主题和长距离依赖
- 多跳问题：多个证据分散在不同区域，扁平 top-k 不容易恢复推理链
- 主题级理解：仅靠底层 chunk 往往很难把握章节、主题、文档整体结构

树状 RAG 的优势在于：

- 同时利用高层摘要节点与底层叶子节点
- 可以在主题级和证据级之间切换粒度
- 更自然地承载多跳、多层级推理路径

因此，本研究默认站在一个前提上：树状组织形式在长文档、多跳、知识密集任务上具有研究价值。

此外，需要补充一个边界判断：第一阶段不把“图结构”直接纳入主线问题定义，而是先把图视为未来可转换为搜索树的一类更一般结构。也就是说，当前主线依然是树状导航；图相关讨论只作为后续应用场景储备，不抢占第一阶段的工程主线。

### 3.2 为什么要研究回溯

现有多步检索和推理系统普遍面临一个问题：一旦早期选错分支，错误上下文就会持续干扰后续判断。

回溯机制的价值在于：

- 允许系统承认“刚才走错了”
- 返回上一个决策点重新选择路径
- 将多步检索从单向前进改造成“试探-评估-修正”的闭环过程

已有很多前沿工作都在做类似事情，例如：

- Self-RAG：通过自我反思决定是否继续检索
- Adaptive-RAG：按问题复杂度动态调整检索策略
- Backtracking Correction：在多步决策链中显式纠错和回退

这些工作说明“修正”和“回退”本身是有价值的；而本课题的切入点是：能否用 Mamba 的状态机制，让这种回退变得更便宜、更稳定、更工程化。

### 3.3 为什么考虑 Mamba 而不是只用 Transformer

Transformer 在检索式导航中通常依赖 KV Cache 保留上下文。问题在于：

- 状态大小随序列长度增长
- 走错分支后的回退成本高
- 深层探索时容易出现显存与重算压力

Mamba/SSM 的关键特征是：

- 历史信息被压缩进固定大小隐状态
- 可以在节点级别保存状态快照
- 理论上支持与路径长度无关的状态恢复成本

因此，Mamba 更适合作为“导航器”而不是“生成器”。

---

## 4. 核心研究问题

本课题建议固定为以下三个核心问题：

### RQ1：机制问题

Mamba 的状态快照与恢复，是否能稳定支撑树状检索中的回溯控制？

### RQ2：系统问题

Navigator-Generator 解耦后，是否可以形成完整、可复现、可审计的闭环系统？

### RQ3：价值问题

即使单步速度不一定优于 Transformer，这种基于状态快照的导航方式，是否在深层探索或资源受限场景下体现系统价值？

---

## 5. 核心主张与边界

### 5.1 本研究的主张

本研究的主张应当控制在以下范围内：

1. 我们提出一种基于状态快照回溯的树状 RAG 导航框架。
2. 该框架将导航与生成明确解耦，使导航过程可记录、可审计、可分析。
3. 该框架的核心贡献位于导航侧状态管理与搜索控制，而不是生成器本身。
4. 在资源受限和深层探索场景下，该方法可能具有系统层面的独特优势。
5. 如果后续实验支持，这套机制还有机会扩展到由实体、事件、时间、地点等关系展开而成的结构化搜索空间。

### 5.2 明确不主张的内容

在第一阶段，不应主张：

- Mamba 全面优于 Transformer
- Mamba 更适合作为答案生成模型
- 本方法已经在所有任务上带来显著准确率领先
- 回溯一定总能提升最终答案质量

这些都只能在后续实验充分支持后，再谨慎表达。

---

## 6. 系统框架定义

整个系统由四个核心模块组成。

### 6.1 Tree Builder

负责把文档或知识单元组织为树结构，包括：

- 节点文本
- 父子关系
- 节点层级信息
- 可选的摘要节点与叶子节点

第一阶段不强求一步做到最复杂的 RAPTOR 风格递归树，但至少要有真实的树状结构，而不是 toy-only 结构。

第二阶段可以进一步考虑一种更贴近真实多跳检索的输入来源：把实体、事件、时间、地点等结构化关系组织成图，再按查询把局部图展开为搜索树输入 Controller。这样既不破坏当前 Tree Builder 主线，也为后续应用场景保留接口。

### 6.2 Navigator

由 Mamba 充当树状检索中的状态更新器，其职责是：

- 顺序读取当前节点文本
- 更新当前隐状态
- 导出可供路由和评估使用的状态表示
- 支持在关键节点保存 snapshot handle

注意：Navigator 不直接生成自然语言答案。

### 6.3 Router + SSGS Controller

这部分是控制核心：

- Router 负责对候选子节点进行打分或排序
- Controller 负责 DFS 风格的探索与回溯
- 当当前路径被判定为低价值时，触发状态恢复
- 记录访问路径、回滚次数、栈深度和失败原因

这里需要明确：

- Mamba 不是决策器本身
- 选路由独立模块承担
- SSGS 是控制逻辑，不是基础编码器

### 6.4 Generator

生成器建议继续采用成熟 Transformer 模型，其输入应严格限定为：

- 最终证据文本列表
- 可选 trace 摘要
- 用户问题

它不继承 Navigator 的内部状态，不感知任何快照栈信息。

---

## 7. 解耦原则

本研究必须坚持以下硬边界：

1. Navigator 只负责导航和证据发现，不负责最终答案生成。
2. Generator 只消费文本证据，不继承 Mamba 内部隐状态。
3. 所有关于生成质量的结论，默认都属于次级结论，除非能严格归因到导航质量变化。

可以把整个系统理解为：

- Mamba 是探路员
- Router 是指南针
- SSGS Controller 是回溯控制器
- Transformer Generator 是写报告的人

---

## 8. 第一阶段最终要拿到什么

第一阶段的目标不是“大规模比较实验”，而是拿到一个真实可用的系统锚点。

### 8.1 交付物

第一阶段至少应交付：

1. 一个从输入问题到输出答案的完整闭环系统
2. 一个支持状态快照保存与恢复的 Navigator
3. 一个支持回溯的树状检索控制流程
4. 一份冻结字段的 trace 输出
5. 一个最小可复现运行脚本

### 8.2 成功标准

第一阶段成功的判据应为：

1. 单条命令可复现实验
2. 至少一个真实样例可完整审计
3. 能显示系统访问了哪些节点
4. 能显示是否发生了 rollback
5. 能输出最终 evidence list
6. 能完成 generator readout
7. 能做最基本的 failure attribution

---

## 9. 建议冻结的实验协议

为了避免后面反复改实验口径，建议尽早冻结以下内容。

### 9.1 Trace schema

第一阶段建议至少记录以下字段：

- `routing_mode`
- `context_source`
- `leaf_indices_required`
- `nav_target_leaf_index`
- `nav_success`
- `visited_leaf_visits_ordered`
- `visited_leaf_indices_deduped`
- `rollback_count`
- `snapshot_stack_max_depth`
- `nav_wall_time_ms`
- `context_build_error`
- `exact_match`
- `rouge_l_f1`

### 9.2 Arm 定义

建议至少保留以下比较臂：

- `oracle_item_leaves`
- `flat_leaf_concat`
- `t1_visited_leaves_ordered`
- `flat_topk_semantic`

其中第一阶段主线先以：

- `oracle_item_leaves`
- `flat_leaf_concat`
- `t1_visited_leaves_ordered`

为主，`flat_topk_semantic` 可作为补充。

### 9.3 写作边界

文中必须区分：

- measured result
- theoretical projection

不能混写。

---

## 10. 第一阶段只做什么，不做什么

### 10.1 第一阶段只做什么

第一阶段只做四件事：

1. 搭完整框架
2. 冻结 trace 与协议
3. 跑出一个可审计的闭环样例
4. 建立最小比较基础

### 10.2 第一阶段不做什么

在框架跑通之前，不建议做：

- OOM 边界大规模实验
- 深度/宽度 sweep
- 复杂 learned router
- 大规模 benchmark 追榜
- Generator 优化和 prompt 花活

这些工作应该放到第二阶段。

这里特别要避免一个写作误区：不要把“实体/事件主动结构化”写成当前项目已经拥有的成熟前序模块。对本项目而言，这一方向更准确的表述是“受相关文献启发的潜在数据构建与应用场景”，不是当前主线已经完成的既有资产；已有的 `EST + 睡眠记忆巩固` 背景也不应被直接等同为本项目已经具备的结构化图构建能力。

---

## 11. 第二阶段再讨论什么

当第一阶段闭环稳定后，再进入 controlled comparison：

第二阶段的核心任务不应再只是继续观察导航行为，而应回答一个更关键的问题：

**导航侧的差异，是否会稳定传导到最终答案质量。**

因此，第二阶段的优先级应固定为：

1. 先做固定生成器的端到端评测
2. 再决定 learned head 是否值得进入主线优化
3. 最后才讨论模型继续放大或更复杂搜索空间

### 11.0 第二阶段的首要目标

第二阶段首先要完成的，不是继续扩充 routing 种类，而是建立下面这条证据链：

`navigation behavior difference -> evidence/context difference -> final answer quality difference`

如果这条链路还没有建立，那么直接主推 learned head 很容易变成“在优化一个尚未证明具有下游价值的中间模块”。

### 11.0.1 为什么先做端到端，而不是先做 learned head

原因主要有三点：

1. 第一阶段已经证明导航框架能跑通
- 当前缺的不是“还能不能再接一个 router”
- 而是“导航差异是否真的影响答案输出”

2. 当前 learned classifier 的收益还没有被证明
- 它已经证明了系统可接入 learned routing arm
- 但尚未显示出足以进入主结果表的稳定优势

3. 端到端评测能帮助决定第二阶段真正瓶颈在哪里
- 如果端到端指标对 routing 差异不敏感，说明继续优化 learned head 的优先级应下降
- 如果端到端指标明显受 evidence 质量影响，再做 learned head 才更有针对性

**与本仓库实验记录对齐（2026-04）**：固定生成器与本机 Qwen、**`context_select` 对齐** 下的 **B 链 500 三连**（`overlap_k4` / `cosine_probe` / Oracle）见 [`Navigation_Experiment_Record_CN.md`](Navigation_Experiment_Record_CN.md) **§9.12**：**Oracle 显著高于 `rule`**，**`cosine_probe` 低于 `rule`**（EM 与 F1 均劣）。因此下文 **§11.0.5** 中「`rule` 与 `cosine_probe` 差异很小」**仅指 §9.8 表内协议**（双臂 **`+ anti_collapse`**），**不**否定其它协议下 cosine 更差的可能。

### 11.0.2 第二阶段的推荐执行顺序

推荐按以下顺序推进：

1. 固定 Generator
- 保持生成器、prompt、解码参数不变
- 不让生成侧变化掩盖导航侧差异

2. 固定上下文构建协议
- 明确 generator 最终消费什么文本
- 保证不同 arm 的输入口径一致

3. 跑一轮端到端对比
- 先用 `100` 到 `200` 条真实子集做干净评测
- 不必一开始就全量扩大

4. 根据端到端结果决定 learned head 是否升级为主线
- 若端到端差异明显，再继续优化 learned head
- 若端到端差异很小，则优先考虑预算协议、context build 或数据难度，而不是继续堆 router

### 11.0.3 第二阶段建议冻结的比较臂

第二阶段第一轮端到端评测建议至少保留以下 arm：

1. `oracle_item_leaves`
- 作为近似上界参考，回答“如果证据给对了，生成器能做到什么水平”

2. `flat_leaf_concat`
- 作为无回溯树导航的弱基线

3. `t1_visited_leaves_ordered`
- 作为当前 SSGS 导航输出的直接消费版本

4. `370M + rule`
- 当前主线 routing 基线之一

5. `370M + cosine_probe`
- 当前主线 routing 基线之二

当前不建议在第二阶段第一轮主表中强行加入：

- `370M + learned_classifier`

更稳妥的做法是先把它放进补充实验，待端到端证据表明“路由继续优化值得做”后，再决定是否把它升格为正式主臂。

### 11.0.4 第二阶段建议冻结的指标

端到端评测建议明确分成三类指标：

1. 最终答案指标
- `exact_match`
- `answer_f1` 或等价 QA 指标
- `rouge_l_f1`

2. 导航过程指标
- `nav_success_rate`
- `avg_nav_wall_time_ms`
- `avg_rollback_count`
- `avg_evidence_count`

3. 归因指标
- `navigation failure`
- `context construction failure`
- `generation failure`

其中需要特别强调：

- 由于当前 `avg_evidence_count` 仍容易打满 budget，它在第二阶段仍应被视为辅助指标
- 在没有解除 budget 饱和前，不应把 evidence 数量差异写成主结论
- **补充（2026-04）**：应用 `scripts/diagnostics/analyze_evidence_saturation.py` 对 `run_payload.json` 批量汇总，区分 **「预算顶满」**、**「证据实体多样性」** 与 **「金叶子是否曾访问 / 是否被接受进 evidence」**，避免仅凭条数下结论

### 11.0.5 第二阶段的判停标准

第二阶段第一轮端到端评测结束后，应先判断是否满足以下任一条件：

1. `rule` 与 `cosine_probe` 的最终答案指标差异很小（**需核对比较协议**：实验记录 **§9.8** 在 **`rule + anti_collapse` vs `cosine + anti_collapse`** 上曾观测到差异很小；**§9.12** 在 **`context_select` 固定、`overlap_k4` vs 纯 `cosine_probe`** 上观测到 **cosine 明显更差**）
- 若确认为 **§9.8 同类协议** 且差异仍很小：说明 **该协议下** routing 差异未明显传导到生成结果，优先检查 context build、budget 与任务难度
- 若协议为 **§9.12**：则不宜用「差异很小」判停；应回到证据发现 / 路由质量与 **§10.1.1** 过程指标

2. `oracle_item_leaves` 明显高于导航臂
- 说明系统瓶颈仍主要在导航/证据发现
- 这时再做 learned head 更有意义

3. `oracle_item_leaves` 与导航臂差距也不大
- 说明生成器或 prompt 可能已成为更主要瓶颈
- 此时应谨慎评估是否需要把工作重心部分转向端到端 readout

### 11.1 比较维度

第二阶段在首轮端到端验证之后，再系统比较：

1. 比较不同 routing 模式
- rule
- cosine probe
- learned classifier

2. 比较不同上下文构建方式
- oracle
- flat
- visited leaves

3. 比较不同资源预算下的表现
- 显存预算
- 最大深度
- 最大回溯次数
- wall-clock latency

4. 讨论系统价值
- 深层探索时是否更稳
- 是否更易审计
- 是否更能区分导航失败与生成失败

### 11.2 模型规模切换顺序

导航器的正式模型切换建议按以下顺序进行，而不是一开始直接上大模型：

1. `smoke 级后端`
- 先确认导航链路、trace、rollback、registry、batch summary 全部跑通

2. `370M`
- 这是第一正式规模
- 适合做第一轮 routing 对比和多样本导航实验
- 也是最适合先观察导航行为是否稳定的规模

3. `1.4B`
- 只在 370M 已经稳定后再切
- 用于验证规模放大后导航是否仍保持收益
- 不建议在 pipeline 尚未稳定时直接用它做主线

原则上：

- `370M` 应该早于 `1.4B`
- Generator 固定为 Qwen，不跟着一起变化
- 先比较导航，再比较模型大小

### 11.3 Learned Head 的接入时机

可学习头应该进入主线，但不应早于第一个正式预训练导航器。

建议顺序如下：

1. `smoke 后端`
- 先保证 pipeline、trace、batch、rollback 全通

2. `固定路由基线`
- 先完成 `rule` 与 `cosine_probe` 的导航对比

3. `370M 预训练导航器`
- 先验证正式预训练表示是否带来更稳定的导航收益

4. `learned head`
- 把可学习头挂在 Router/Scoring 模块
- 输入为 query 表示、当前状态摘要、候选子节点表示等
- 输出为候选 child score 或 relevance score
- 推荐先在 370M 上训练轻量线性/MLP 评分头，而不是一开始就和 1.4B 同时上

5. `1.4B`
- 最后再做规模放大

因此，可学习头是主线的一部分，但不是下一步最先要做的内容。

截至当前真实 `2Wiki` 子集实验，可学习头已经完成系统接入与真实语料运行验证，但其表现应谨慎定位：它证明了框架能够容纳 learned routing arm，却尚未证明当前 learned head 已经成为有效主方案。当前更稳妥的写法是把 learned classifier 作为补充实验或负面案例保留，用来说明系统即使在较差路由下仍具有可运行性与可审计性，而不是把它写成当前阶段的主结果。

### 11.4 实体/事件结构化图作为后续应用场景

当第一阶段树状导航闭环已经稳定后，可以考虑引入一个更具挑战性的应用场景：基于实体、事件、时间、地点等关系构建的结构化图导航。

这个方向值得保留，原因不是它听起来更大，而是它更贴近 SSGS 机制真正可能体现价值的场景：

1. 多跳路径更深
- 一个问题可能需要跨越多个实体和事件节点才能找到证据链

2. 分支更多，更容易走错
- 从一个人物、地点或时间节点出发，通常会遇到大量候选边和邻接节点
- “试探一条边再回退”的需求会比普通浅层树更频繁

3. 更适合观察 rollback 的系统价值
- 当错误分支很多时，状态快照和恢复是否真的能降低试错成本，会更容易被观察到

但这里必须保持严格边界：

- 当前主线仍是树状 RAG，不直接切换成原生图导航项目
- 在写作上，更稳妥的表述是 `entity-structured graph` 或 `event-centric graph`
- 不要过早宣称已经构建了标准知识图谱，也不要默认拥有完整 ontology、实体链接和图质量评估体系
- 工程实现上，优先选择“图按查询展开为搜索树”的接入方式，而不是立刻重写原生 graph controller

这部分与当前项目已有基础之间需要明确切分：你此前的研究背景是 `EST + 睡眠记忆巩固`，而这里讨论的实体/事件结构化图导航，更多是从相关文献中吸收来的潜在扩展场景，而不是你已经完成的主动结构化系统。因此，在论文、开题和阶段汇报中，应将其定位为“第二阶段扩展场景”而非“既有成熟基础”。

---

## 12. 当前建议的论文叙事

当前最合适的论文叙事不是“纯效果论文”，也不是“纯理论论文”，而是：

平衡型叙事：以系统机制为主，以有效性验证为辅。

可以按如下结构组织：

### 12.1 Introduction

- 树状 RAG 在长文档、多跳任务上的潜力已被验证
- 但树状导航过程缺少低成本回溯机制
- 现有回溯/修正方法大多基于 Transformer，工程代价较高
- 我们提出基于 Mamba 状态快照的树状导航框架

### 12.2 Method

- Navigator-Generator decoupling
- State snapshot and restore
- Routing module
- SSGS controller
- Trace schema and auditing interface

### 12.3 Experiments

- 先证明系统闭环可运行
- 再证明协议下可比较
- 最后讨论在资源受限场景下的系统价值
- 如果主线实验稳定，可增加一个 `entity-structured graph` 应用节，展示该机制在高分支、多跳、频繁回退场景中的潜在价值；但应明确该节属于后续扩展验证，而不是第一阶段主结论

### 12.4 Discussion

- 单步速度可能并不占优
- 但在深层探索与回溯频繁的情况下，状态管理方式可能带来独特优势
- 解耦架构还有助于减少错误上下文向生成器泄漏

---

## 13. 项目归档与仓库组织原则

项目从一开始就不应简单平铺。建议把“研究文档、代码、实验配置、数据、输出结果”分开归档，避免后期文件爆炸后失控。

建议遵守以下原则：

1. 根目录只放入口文件和全局说明。
2. 研究文档统一进入 `docs/`。
3. 可执行代码统一进入 `src/`。
4. 运行脚本与实验脚本分离，统一进入 `scripts/`。
5. 配置文件进入 `configs/`。
6. 原始数据、中间数据、缓存、实验输出分层保存，不混放。
7. 跑出来的结果不要直接散落在根目录，统一进入 `outputs/`。

推荐目录树如下：

```text
mamba2.1/
├─ README.md
├─ .gitignore
├─ docs/
│  ├─ research/
│  │  └─ SSGS_Research_Framework_CN.md
│  ├─ meetings/
│  ├─ papers/
│  └─ notes/
├─ configs/
│  ├─ data/
│  ├─ model/
│  └─ experiment/
├─ scripts/
│  ├─ build_tree/
│  ├─ run_nav/
│  ├─ run_eval/
│  └─ utils/
├─ src/
│  ├─ tree_builder/
│  ├─ navigator/
│  ├─ router/
│  ├─ controller/
│  ├─ generator_bridge/
│  ├─ tracing/
│  └─ evaluation/
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

---

## 14. Git 仓库建议

这个项目非常适合尽早放到 Git 仓库里管理，但应遵守“代码与文档进仓、重量级数据与运行产物不进仓”的原则。

建议纳入 Git 的内容：

- `docs/`
- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `README.md`
- 轻量级示例配置和小样本数据说明

建议不要直接纳入 Git 的内容：

- 大体积原始数据
- 模型权重
- 本地缓存
- 临时输出
- 实验日志大文件
- GPU 中间产物

建议后续采用如下节奏：

1. 先把仓库结构搭好。
2. 再初始化 Git。
3. 第一批提交只包含文档、目录结构、配置模板和最小代码骨架。
4. 数据与输出目录通过 `.gitignore` 管理。

如果后期需要远程协作或换机器复现，这种结构会非常省事。

---

## 15. 当前阶段完成状态

### 15.1 已完成的核心内容

截至当前阶段，下面这些内容可以认为已经完成：

1. 完整导航闭环框架已经搭成
- 包括 `Tree Builder / Navigator / Router / Controller / Generator Bridge / Trace`

2. Mamba 状态快照与回溯链已经可运行
- 相关事件、回溯计数、批次信息、registry 与 summary 都能稳定落盘

3. 真实数据入口已经打通
- 已支持 `2Wiki -> wiki-longdoc -> corpus/qa -> tree payload + batch manifest` 的完整预处理链

4. 正式预训练导航器已经接入
- `state-spaces/mamba-370m-hf` 已在服务器上完成真实子集批量运行

5. 批量运行实现已经完成关键性能收口
- 同一 batch 内共享 controller，避免每个样本重复加载预训练导航器

6. 证据与归因诊断工具已落地（见 **16.3**）
- 支持对「预算饱和 vs 金叶子未达」做批量统计，服务后续导航策略迭代

因此，从“导航部分的基础框架是否已经立住”这个问题看，答案可以写成：**是，已经基本完成。**

### 15.2 当前可稳说的 measured results

当前最稳妥、最值得固定的 measured result 来自 `2Wiki` 真实子集的主线比较：

1. 在 `500` 条真实子集上，`370M + rule` 与 `370M + cosine_probe` 都达到 `nav_success_rate = 1.0`
2. 两种 routing 的行为差异主要体现在搜索形态，而不是是否能完成导航
- `rule` 的平均回溯次数更高，说明其搜索更激进、更频繁试探
- `cosine_probe` 的平均回溯次数更低，说明其搜索更保守
3. 两组平均耗时处于同一量级，但耗时排序在不同规模实验中有一定波动
- 因此，当前阶段不应把“谁绝对更快”写成过强结论

这里必须明确一个写作边界：

- 两组的 `avg_evidence_count` 在当前主线实验里持续打满 evidence budget
- 这意味着 evidence 数量仍然受预算上限约束
- 因此，现阶段**不要把 evidence 数量差异写成主结论**

补充（2026-04，`pilot200` 导航诊断）：

1. `never_visit_gold` 仍是第一瓶颈（约 `57%~61%`）。  
2. 对 `never_visit_gold` 做 root 归因后，`root-miss` 约占 **98.4%**，`in-root-miss` 约 **1.6%**。  
3. 启发式改动（`cos/probe/pool`）可带来局部改善，但离 Oracle 仍有明显差距；启发式阶段应收口并切换学习式 root 决策主线。

### 15.3 当前阶段对 learned head 的定位

当前 **全树线性 `learned_classifier`** 的系统地位可以固定为：

1. 已完成接入
- 代码、训练脚本、checkpoint 与真实语料运行链都已打通

2. 已完成真实语料验证
- 它不再只是概念接口，而是真正在真实子集上跑过的一条 routing arm

3. 单独作为主 routing 仍不成立
- 历史实验显示回溯与证据形态问题仍在；**纯 learned** 不适合无约束主推

**补充（2026-04）**：**`learned_root_classifier` + `learned_root_blend_alpha`**（根上与 `RuleRouter` 分数混合）在 **`500`** 上已恢复与冻结启发式同量级的金叶过程指标与可接受的终点 EM（见 **`Navigation_Experiment_Record_CN.md` §6.5**）。叙事上应区分 **「纯 learned」** 与 **「混合 root」**，后者可作为当前阶段 **有限主线的工程选项**，并固定 **`α` 与训练导出 cap**。

---

## 16. 实验记录与后续计划

### 16.1 实验记录文档

建议把当前阶段的关键实验单独归档，形成可追溯记录。建议文档：

- `docs/research/Navigation_Experiment_Record_CN.md`

其中应固定记录：

- 数据来源与抽样规模
- 配置文件
- 关键 batch id
- measured result
- 当前阶段的可说结论与不可说结论

### 16.2 后续计划

当前最合理的后续计划不是继续无节制扩展模块，而是按下面顺序推进：

1. 冻结当前 `500` 条主线结果
- 主表先固定为 `370M + rule` 与 `370M + cosine_probe`

2. 固定 `small50` 端到端 2x2 消融结论
- `postprocess` 在当前设置下收益为 `0`
- `anti_collapse` 带来可复现提升（`EM/F1` 同步提高）
- 因此下一轮默认采用 `rule + anti_collapse`，`postprocess` 仅保留为可审计开关

3. 整理结果表和实验叙事
- 把成功率、耗时、回溯行为差异，以及 2x2 消融结论写成阶段性结论

4. learned head 从“补充实验”升级为下一主线（限定 root 层）
- 先做学习式 root 路由（直接打 `root-miss`），停止继续扩启发式大网格

5. 后续扩展再分层推进
- 如有需要，再讨论 `1.4B`
- 再讨论更严格的预算实验或 OOM 边界实验
- 最后再讨论结构化图应用扩展

### 16.4 2026-04 阶段收口与下一阶段入口

基于 `pilot200` 对照，当前启发式阶段收口为：

1. **启发式工作基线（冻结）**：`router_cosine_weight=0.7 + probe=1 + max_evidence=8 + context_select_pool_max_items=20`。  
2. **不再继续大规模启发式扫参**：硬 top-M 与激进 probe 已验证存在 `B3->B1` 回退风险。  
3. **正式切换学习式主线**：root 层候选排序学习（目标优先降低 `root-miss`），再验证对 `in_context` 与 EM/F1 的传导。

学习式阶段最小验收门槛：

- `never_visit_gold` 下降；  
- `in_context` 上升；  
- `B3->B1 <= B1->B3`；  
- 在可接受 `nav_ms` 增量下成立（同预算对照）。

**工程结论（2026-04）**：在 **全 fan-out** 下 **纯线性 root 头**不可用；**`learned_root_blend_alpha`**（与 rule 混合）为当前可行形态；**`500` 复验后默认 `α=0.5`**（相对 `0.25` 金叶同量级、**`nav_ms` 更优**）。小样本扫参时用 **`run_navigation_batch.py --max-samples N`** 保持 **同一 manifest 与同一 checkpoint**，只调 **`α`** 与 **`batch_id_prefix`**（见 **`Navigation_Experiment_Record_CN.md` §6.5**）。

### 16.3 工程与运维增量（2026-04，与论文叙事并行）

下列内容不改变 RQ 表述，但影响**可复现性、归因清晰度与线上排障**，建议在实验记录中随手标注「当时代码版本」。

**Phase A 管线（`run_navigation_sample`）**

- 树 JSON **单次读取**：`load_tree_payload` + `load_tree_from_payload`，避免重复 I/O；叶子索引映射单次构建、上下文构建复用。
- **上下文构建失败**时跳过生成器调用，并写入明确 `generation_error`，避免空上下文仍走生成。
- **`eval_mode`**（`generation` / `retrieval`）与 **`report_dir`** 写入 payload 与 `run_registry.jsonl`，便于区分打分对象与多实验目录隔离。
- **实验模版默认值**：`configs/experiment/` 下真实语料相关 JSON 对 `t1_visited_leaves_ordered` / `flat_leaf_concat` 已写入 `context_select_mode=question_overlap_topk`、`context_select_k=4`（2026-04-18 bump，与服务器 B 全量结论一致）；`oracle_item_leaves` 例题为 `context_select_mode=off`（代码层未传键时仍为 `off`，见专档 MI-006）。

**Navigator（`Mamba2Navigator`）**

- 问题侧向量 **LRU 有界缓存**、`clear_cache()`；可选 **`use_ssm_continuity`**（Phase 1 默认关闭）及对 `reset_state`/`reset_cache` 的探测式清理，与快照栈策略对齐文档边界。
- 多后端 **last hidden** 抽取统一为 `_extract_last_hidden`，形状异常时 fail-fast。

**诊断脚本**

- `scripts/diagnostics/analyze_evidence_saturation.py`：从 `run_registry.jsonl`（按 `batch_id`）或 `glob` 加载 `run_payload.json`，输出证据预算饱和率、证据内实体多样性、金叶子访问与接受情况（依赖 batch 传入 `positive_leaf_indices` → trace `leaf_indices_required`）。
- 增强项：支持 `--with-context-gold-metrics`，可对 `generator_evidence_texts/context_texts` 计算 gold 文本覆盖率，用于区分“导航拿到证据”与“生成器实际看到证据”。

**Readout-first、判停规则、服务器同步与 `context_select_mode` 开关的完整说明**

- 上述主题的**现象、根因与处置**仅维护于 [`docs/Major_Issues_And_Resolutions_CN.md`](../Major_Issues_And_Resolutions_CN.md)（**MI-004 ~ MI-006、MI-002**）。本节其余条目仍为组件级能力说明，不与专档重复。

---

## 17. 一句话结论

现阶段最重要的事情已经不是“证明我们已经比谁都强”，而是确认下面这件事已经真实成立：

我们已经做出一个真实可运行的、树状的、可回溯的、可审计的、导航与生成解耦的 RAG 导航框架，并且它已经在真实公开数据子集上完成了正式预训练导航器实验。

只要这个阶段性锚点成立，后续论文写作、比较实验和系统扩展就都有了可靠起点。
