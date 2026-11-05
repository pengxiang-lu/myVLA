# myVLA

## 一、简介

　　**myVLA 是一个基于 PyTorch 的轻量视觉—语言—动作（VLA）项目**，支持 LIBERO 和 MetaWorld MT50 上的行为克隆训练与闭环评估。模型使用冻结的 CLIP 编码图像和语言，通过可训练的动作解码器预测连续动作块。

　　项目包含 V1 基础版和 V2 双向融合版。V2 在动作解码前增加视觉—语言交叉注意力，下图为其 LIBERO 配置。

![myVLA V2 模型架构](assets/v2.png)

## 二、安装教程

### 2.1 安装依赖

　　预先安装 `uv`，并在项目根目录执行以下命令。

​	默认使用 Python 3.10，依赖版本由 `uv.lock` 固定。

​	LIBERO 和 MetaWorld 的 MuJoCo 版本不同，分别使用独立虚拟环境，不能同时启用 `sim` 和 `mt50`。先用下面的 `uv sync` 安装对应依赖，后续统一使用 `uv run 脚本路径`；`UV_PROJECT_ENVIRONMENT` 决定使用哪个虚拟环境。

```bash
# LIBERO 环境：.venv
unset UV_PROJECT_ENVIRONMENT
uv sync --locked --extra sim
uv run scripts/doctor.py

# MetaWorld 环境：.venv-mt50
UV_PROJECT_ENVIRONMENT=.venv-mt50 uv sync --locked --extra mt50
```

### 2.2 准备 CLIP 权重

　　两套环境共用同一份 CLIP 权重。下载脚本支持分段续传及校验，默认保存到 `models/clip-vit-base-patch16/`。

```bash
unset UV_PROJECT_ENVIRONMENT
uv run scripts/download_clip.py --workers 4
```

　　两份配置均通过相对路径 `models/clip-vit-base-patch16` 加载本地权重，因此命令应在项目根目录运行；权重放在其他位置时修改 `model.backbone`。`HF_HUB_OFFLINE=1` 可选，用于禁止联网检查。

​	正式训练需要可用的 CUDA；CPU 调试需同时设置 `train.device: cpu` 和 `train.precision: fp32`。

### 2.3 准备 LIBERO 源码

　　LIBERO 闭环评估需要源码和仿真资源。没有源码时，将官方仓库克隆到默认查找位置；已有源码可跳过克隆，通过 `LIBERO_ROOT` 指定目录。

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
export LIBERO_ROOT="$PWD/third_party/LIBERO"
export MUJOCO_GL=egl
uv run scripts/sim_smoke.py
```

　　本机已有源码可使用 `export LIBERO_ROOT=/home/lpx/python-project/LIBERO`。仿真依赖由本项目管理，无需额外安装上游整份 requirements；仅训练 HDF5 数据时不需要启动模拟器。

## 三、训练

　　直接运行 `src/myvla/train.py` 训练、`src/myvla/evaluate.py` 评估，参数跟在脚本名后。在项目根目录使用下面的完整路径；在 `src/myvla/` 目录也可运行 `uv run train.py ...` 或 `uv run evaluate.py ...`。脚本会切换到项目根目录，相对的配置、数据、模型及输出路径均以项目根目录解释。

### 3.1 LIBERO 数据准备

```bash
unset UV_PROJECT_ENVIRONMENT
uv run scripts/download_data.py \
  --suite libero_spatial --directory data

uv run scripts/inspect_data.py \
  --data-dir data/libero_spatial
```

　　下载中断后可重跑同一命令续传。网络需要时，可追加 `--endpoint https://hf-mirror.com --max-workers 2`。已有原始 LIBERO HDF5 数据可直接通过配置中的 `data.directory` 指定路径。

### 3.2 LIBERO 模型训练

　　V1 使用本地 CLIP 配置进行 30,000 步训练，输出到新目录：

```bash
uv run src/myvla/train.py \
  --config configs/train_cfg_v1.yaml \
  --steps 30000 --output-dir outputs/train/libero_spatial_v1
```

　　V2 使用双向融合配置开始独立训练，不加载 V1 动作头。`configs/` 只保留 `v1.yaml` 和 `v2.yaml`，两份配置均默认训练 30,000 步：

```bash
uv run src/myvla/train.py \
  --config configs/train_cfg_v2.yaml \
  --steps 30000 --output-dir outputs/train/libero_spatial_v2
```

　　两份配置的学习率、数据增强、batch、梯度累积及训练计划一致，仅融合层数和默认输出目录不同。先检查流程时，可将步数改为 `--steps 200` 并使用单独输出目录。训练步数按优化器更新尝试计数，不是 epoch。

### 3.3 MetaWorld MT50 数据准备与训练

　　切换到 MT50 环境，下载官方 LeRobot v3 图像/parquet 数据，并导入为训练使用的 HDF5。完整原始数据约 45.9GB，还需预留转换后的空间；将 `/path/to/datasets` 改为实际下载位置。

```bash
export UV_PROJECT_ENVIRONMENT=.venv-mt50

uv run scripts/download_data.py \
  --suite metaworld_mt50 --directory /path/to/datasets --max-workers 2

uv run scripts/convert_metaworld.py \
  --source /path/to/datasets/metaworld_mt50_raw \
  --output data/metaworld_mt50

uv run scripts/inspect_data.py \
  --dataset metaworld --data-dir data/metaworld_mt50

HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --config configs/v2.yaml --dataset metaworld
```

　　导入目标目录必须不存在。当前导入器支持 LeRobot v3 内嵌 RGB，不支持旧版 v2 或视频存储。`--dataset metaworld` 自动切换相机数量、状态/动作维度和默认数据目录，V1、V2 均可使用；数据在其他位置时追加 `--data-dir`。上述 V2 命令默认训练 30,000 步，输出为 `outputs/train/metaworld_mt50_v2`；其输入输出维度与 LIBERO 不同，需要独立训练。

### 3.4 参数设置与断点恢复

　　查看训练参数使用 `uv run src/myvla/train.py --help`。两份配置默认使用 LIBERO-Spatial，输出分别为 `outputs/train/libero_spatial_v1` 和 `outputs/train/libero_spatial_v2`；命令行 `--output-dir` 可以覆盖。主要训练参数在 YAML 中设置：`lr` 为学习率，`steps` 为总步数，`batch_size` 为单批样本数，`grad_accumulation` 为梯度累积次数。单卡有效 batch 为两者之积；`image_shift` 和 `color_jitter` 控制训练期间的图像增强。

　　训练结果统一保存到 `outputs/train/<实验名>/`，评估结果保存到 `outputs/eval/<实验名>/`。训练目录保存 `latest.pt`、`best.pt` 和 `metrics.jsonl` 等文件。`latest.pt` 用于恢复训练，`best.pt` 对应最低离线验证 loss；只会恢复到已保存的步数。仅传 `--resume` 时，默认继续写入 checkpoint 所在目录，也可通过 `--output-dir` 指定其他目录。

```bash
# 恢复 LIBERO 训练
unset UV_PROJECT_ENVIRONMENT
HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --resume outputs/train/libero_spatial_v1/latest.pt

# 恢复 MT50 训练
export UV_PROJECT_ENVIRONMENT=.venv-mt50
HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --resume outputs/train/metaworld_mt50_v2/latest.pt
```

　　`--resume` 恢复完整训练状态，要求架构、数据和训练计划等保持一致；`--init-checkpoint` 用于加载权重开始新的微调，并重置优化器，两者不能同时使用。已有输出目录会被保护，重新训练请设置新的 `--output-dir`。

## 四、评估

　　`evaluate.py` 会根据 checkpoint 自动选择 LIBERO 或 MT50；也可用 `--dataset libero` 或 `--dataset metaworld` 显式核对。查看完整参数使用 `uv run src/myvla/evaluate.py --help`。

### 4.1 LIBERO 闭环评估

```bash
unset UV_PROJECT_ENVIRONMENT
export LIBERO_ROOT=/home/lpx/python-project/LIBERO

HF_HUB_OFFLINE=1 MUJOCO_GL=egl uv run src/myvla/evaluate.py \
  --checkpoint outputs/train/libero_spatial_v2/best.pt \
  --suite libero_spatial --episodes 10 --execute-steps 4 \
  --output-dir outputs/eval/libero_spatial_v2
```

　　将 `LIBERO_ROOT` 和 `--checkpoint` 改为实际路径。省略 `--tasks` 会评估整个套件；追加 `--tasks 0` 可仅评估单任务，追加 `--videos` 保存视频。Spatial 默认每任务 10 回合、每回合最多 220 个控制步，每次执行 4 步预测动作后重新观察。

### 4.2 MetaWorld MT50 闭环评估

```bash
export UV_PROJECT_ENVIRONMENT=.venv-mt50

HF_HUB_OFFLINE=1 MUJOCO_GL=egl uv run src/myvla/evaluate.py \
  --checkpoint outputs/train/metaworld_mt50_v2/best.pt \
  --episodes 50 --execute-steps 1 --protocol lerobot \
  --output-dir outputs/eval/metaworld_mt50_v2
```

　　默认评估 50 个任务，每任务 50 回合，每回合最多 500 步。`--protocol lerobot` 使用可复现的随机位置；`--protocol benchmark` 使用 MetaWorld MT50 提供的 goal，每任务最多 50 个。快速检查可追加 `--tasks 0 --episodes 2 --horizon 50`，保存视频使用 `--videos`。

　　评估输出为 `episodes.jsonl`、`results.json` 和可选 MP4，根据环境成功信号统计成功率。离线验证 loss 不等于成功率；比较实验时应保持评估协议、seed、回合数及动作执行长度一致。每次评估使用新的输出目录。
