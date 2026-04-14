# SSGS 双环境配置说明

## 1. 环境分工

当前项目建议明确采用双环境策略：

- 本地 Windows + conda：用于开发、协议验证、轻量 smoke test
- AutoDL Linux 服务器：用于正式导航实验、批量运行与后续真实 `mamba_ssm` 集成

生成端先固定为 `Qwen`，避免在导航器尚未稳定时引入多余变量。

---

## 2. 当前配置文件

本地 native + Qwen：

- `configs/experiment/local_native_qwen.json`

服务器 mamba_ssm + Qwen：

- `configs/experiment/server_mamba_ssm_qwen.json`

模型模板：

- `configs/model/mamba2.example.json`
- `configs/model/qwen.example.json`

---

## 3. 本地环境说明

### 3.1 当前已确认状态

当前已经实际探测过两个本地解释器：

1. Cursor 当前默认 `py` 指向的解释器
2. 你指定的 conda 环境 `mamba2`

当前结论如下：

- `torch` 可用
- `transformers` 可用
- `mamba_ssm` 未检测到
- `causal_conv1d` 未检测到
- `mamba2` / `mamba2_native` / `mamba` 这些模块名也未检测到

这说明：

- 现在这个 `mamba2` 环境里还没有可直接导入的 mamba 导航后端
- 或者你之前使用 native 的真实模块名并不是当前配置里的名字
- 或者你之前运行成功时使用的是另一个环境

### 3.2 本地下一步建议

因此，本地下一步应优先做的是：

```powershell
& "C:\Users\26433\miniconda3\Scripts\conda.exe" env list
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" -m pip list
& "C:\Users\26433\miniconda3\envs\mamba2\python.exe" -c "import pkgutil; print(sorted([m.name for m in pkgutil.iter_modules() if 'mamba' in m.name.lower()]))"
```

如果你确认 native 的真实模块名不是 `mamba2_native`，请把对应配置文件里的：

- `navigator_dependency_module`

改成实际可导入模块名。

---

## 4. AutoDL 服务器环境搭建命令

下面给你一套可以直接在 Linux 终端执行的命令。建议先新建环境，不要污染系统 Python。

### 4.0 一次性执行版

如果你想直接复制整段到服务器终端，可以先用下面这版。默认假设：

- 服务器上已安装 `conda`
- 新环境名为 `ssgs`
- 代码仓库目录名为 `mamba2.1`
- 导航器使用 `mamba_ssm`
- 生成端固定为 `Qwen`

```bash
nvidia-smi
python --version
conda --version
nvcc --version || true

conda create -n ssgs python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ssgs

python -m pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio || true
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install transformers accelerate sentencepiece datasets pytest
pip install causal-conv1d
pip install mamba-ssm

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import transformers; print(transformers.__version__)"
python -c "import mamba_ssm; print('mamba_ssm ok')"
python -c "import causal_conv1d; print('causal_conv1d ok')"

git clone https://github.com/keyiadiannao/mamba2.1.git
cd mamba2.1

python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_nav/run_phase_a_pipeline.py --config configs/experiment/server_mamba_ssm_qwen.json
```

如果你还没有把代码推到远程仓库，就先把项目上传后再执行 `git clone`。

### 4.1 基础检查

```bash
nvidia-smi
python --version
conda --version
nvcc --version || true
```

### 4.2 创建并激活环境

```bash
conda create -n ssgs python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ssgs
python --version
```

### 4.3 安装基础依赖

```bash
python -m pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio || true
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install transformers accelerate sentencepiece datasets
pip install pytest
```

### 4.4 安装 `mamba_ssm`

```bash
pip install causal-conv1d
pip install mamba-ssm
```

如果这里报编译或 CUDA 相关错误：

1. 先保存完整报错
2. 不要连续乱装多个版本
3. 先确认 `torch` 与 CUDA 是否匹配
4. 再决定是否需要换 wheel、降版本或改环境

### 4.5 验证依赖

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import torch; print(torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
python -c "import mamba_ssm; print('mamba_ssm ok')"
python -c "import causal_conv1d; print('causal_conv1d ok')"
```

### 4.6 拉代码并进入项目

如果你已经上传到 Git 远程仓库：

```bash
git clone https://github.com/keyiadiannao/mamba2.1.git
cd mamba2.1
```

如果你是手动上传项目文件，则直接进入项目目录：

```bash
cd /path/to/mamba2.1
```

### 4.7 运行测试和示例

```bash
python -m unittest discover -s tests -p "test_*.py"
python scripts/run_nav/run_phase_a_pipeline.py --config configs/experiment/server_mamba_ssm_qwen.json
```

---

## 5. 推荐的服务器排错顺序

如果服务器跑不起来，建议按这个顺序查：

1. `torch.cuda.is_available()` 是否为 `True`
2. `mamba_ssm` 是否可导入
3. `causal_conv1d` 是否可导入
4. 项目测试是否通过
5. Phase A 配置脚本是否能启动

不要一上来就直接跑大实验。

---

## 6. 你这次报错的直接修复办法

你这次服务器报错的核心是：

- 系统检测到的 CUDA 工具链版本是 `12.4`
- 当前安装的 PyTorch 是 `2.11.0+cu130`
- `mamba-ssm` 需要编译扩展时，发现 `12.4 != 13.0`，因此失败

也就是说，问题不是项目代码本身，而是服务器里的 `torch` CUDA 版本装错了。

建议直接在服务器当前环境里执行下面这组修复命令：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ssgs

python -m pip uninstall -y mamba-ssm causal-conv1d torch torchvision torchaudio
python -m pip cache purge

python -m pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install transformers accelerate sentencepiece datasets pytest
pip install causal-conv1d
pip install mamba-ssm

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import mamba_ssm; print('mamba_ssm ok')"
python -c "import causal_conv1d; print('causal_conv1d ok')"
```

如果这组命令执行成功，再继续：

```bash
cd /root/autodl-tmp/mamba2.1
python -m unittest discover -s tests -p "test_*.py"
python scripts/run_nav/run_phase_a_pipeline.py --config configs/experiment/server_mamba_ssm_qwen.json
```

### 6.1 如果依然报 `12.4 != 13.0`（你当前就是这个情况）

这通常不是你主环境的问题，而是 `pip` 的 build isolation 临时环境又拉了不匹配的 `torch`（常见是 `cu130`）。

也就是说：

- 你的主环境里可能已经是 `torch 2.6.0+cu124`
- 但编译 `mamba-ssm` 时，隔离环境里用了另一个 `torch`
- 最终触发 `CUDA mismatch`

针对这个问题，直接执行下面这组命令：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ssgs

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.is_available())"

python -m pip uninstall -y mamba-ssm causal-conv1d
python -m pip install -U pip setuptools wheel ninja packaging

export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8

python -m pip install --no-build-isolation causal-conv1d
python -m pip install --no-build-isolation mamba-ssm

python -c "import mamba_ssm; print('mamba_ssm ok')"
python -c "import causal_conv1d; print('causal_conv1d ok')"
```

如果你的服务器不是 CUDA 12.4，请把 `CUDA_HOME` 改成对应目录。

### 6.2 为什么 `--no-build-isolation` 有效

`--no-build-isolation` 会强制构建过程使用你当前 conda 环境中的依赖，而不是临时创建一个新环境去拉新版本包。  
对于你这个“主环境 cu124，但临时环境拉了 cu130”的问题，这通常是最关键的一步。

---

## 7. 什么时候正式切到真实 Mamba2

建议分两步：

1. 结构上已经切入
- 当前代码已经支持 `navigator_type = mamba2_native` 和 `navigator_type = mamba_ssm`

2. 运行上再切入
- 本地先确认 native 环境里的真实模块名
- 服务器先确认 `mamba_ssm` 依赖完整安装
- 然后再补 `Mamba2Navigator.step()` 的真实前向逻辑

也就是说：

现在已经完成“架构级正式引入”，下一步进入“运行级正式接入”。

---

## 8. 当前与你相关的最短操作路径

如果你现在就要开始，建议按这个顺序做：

1. 先把当前代码推到远程 Git 仓库
2. 在 AutoDL 上按本文件第 4 节新建 `ssgs` 环境
3. 安装 `torch + transformers + causal-conv1d + mamba-ssm`
4. 拉代码并运行：

```bash
git clone https://github.com/keyiadiannao/mamba2.1.git
cd mamba2.1
python -m unittest discover -s tests -p "test_*.py"
python scripts/run_nav/run_phase_a_pipeline.py --config configs/experiment/server_mamba_ssm_qwen.json
```

5. 如果这些都通过，再开始真实 `mamba_ssm` 导航器接入
