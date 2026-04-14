# `configs/` 配置目录约定

- `data/`: 数据集路径、字段映射、采样范围
- `model/`: Mamba、embedding、generator 等模型配置
- `experiment/`: 单次实验的预算、路由模式、输出路径

建议后续把实验差异尽量写进配置文件，而不是散落在代码常量里。
