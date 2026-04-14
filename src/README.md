# `src/` 代码模块约定

- `tree_builder/`: 文档树构建、节点组织、层级结构生成
- `navigator/`: Mamba/SSM 导航器与状态更新
- `router/`: 子节点打分、排序与候选选择
- `controller/`: SSGS 搜索控制、回溯与停止条件
- `generator_bridge/`: evidence 到生成器 prompt 的桥接
- `tracing/`: trace schema、运行记录、审计输出
- `evaluation/`: 指标计算与结果汇总

建议后续新增代码优先放入对应模块，而不是直接堆在 `src/` 根目录。
