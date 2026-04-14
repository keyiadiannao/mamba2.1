# Mamba Tree-RAG / SSGS

本项目用于研究一种面向树状 RAG 的导航框架：使用 Mamba/SSM 作为 `Navigator`，通过状态快照与回溯实现低成本树搜索，并与 Transformer `Generator` 解耦。

当前阶段目标不是先追求大规模榜单结果，而是先搭建一个可运行、可审计、可复现的最小闭环系统。

## 当前研究主线

- 研究对象：树状 RAG 中的导航机制
- 核心机制：状态快照、回溯控制、导航-生成解耦
- 第一阶段目标：完整闭环框架与 trace 审计

详细研究说明见：

- `docs/research/SSGS_Research_Framework_CN.md`

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
