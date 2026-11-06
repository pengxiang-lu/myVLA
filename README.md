# myVLA

## 一、简介

　　**myVLA 是一个基于 PyTorch 的轻量视觉—语言—动作（VLA）项目**，支持 LIBERO 和 MetaWorld MT50 上的行为克隆训练与闭环评估。模型使用冻结的 CLIP 编码图像和语言，通过可训练的动作解码器预测连续动作块。

　　项目包含 V1 基础版和 V2 双向融合版。V2 在动作解码前增加视觉—语言交叉注意力，当前训练配置使用 6 层双向融合块，动作解码器为 4 层。下图展示其结构，图中的层数对应早期 2 层融合版本。

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

　　V1 使用本地 CLIP 配置进行 20,000 步训练，输出到新目录：

```bash
uv run src/myvla/train.py \
  --config configs/train_cfg_v1.yaml \
  --steps 20000 --output-dir outputs/train/libero_spatial_v1
```

　　V2 使用双向融合配置开始独立训练，不加载 V1 动作头。`configs/` 只保留 `v1.yaml` 和 `v2.yaml`，两份配置均默认训练 20,000 步：

```bash
uv run src/myvla/train.py \
  --config configs/train_cfg_v2.yaml \
  --steps 20000 --output-dir outputs/train/libero_spatial_v2
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
  --config configs/train_cfg_v2.yaml --dataset metaworld
```

　　导入目标目录必须不存在。当前导入器支持 LeRobot v3 内嵌 RGB，不支持旧版 v2 或视频存储。`--dataset metaworld` 自动切换相机数量、状态/动作维度和默认数据目录，V1、V2 均可使用；数据在其他位置时追加 `--data-dir`。上述 V2 命令默认训练 30,000 步，输出为 `outputs/train/metaworld_mt50_v2`；其输入输出维度与 LIBERO 不同，需要独立训练。

### 3.4 参数设置与断点恢复

　　查看训练参数使用 `uv run src/myvla/train.py --help`。两份配置默认使用 LIBERO-Spatial，输出分别为 `outputs/train/libero_spatial_v1` 和 `outputs/train/libero_spatial_v2`；命令行 `--output-dir` 可以覆盖。主要训练参数在 YAML 中设置：`lr` 为学习率，`steps` 为总步数，`batch_size` 为单批样本数，`grad_accumulation` 为梯度累积次数。单卡有效 batch 为两者之积；`image_shift` 和 `color_jitter` 控制训练期间的图像增强。

　　训练结果统一保存到 `outputs/train/<实验名>/`，评估结果保存到 `outputs/eval/<实验名>/`。训练目录保存 `latest.pt`、`best.pt` 和 `metrics.jsonl` 等文件。`latest.pt` 用于恢复训练，`best.pt` 对应最低离线验证 loss；只会恢复到已保存的步数。仅传 `--resume` 时，默认继续写入 checkpoint 所在目录，也可通过 `--output-dir` 指定其他目录。

　　V1、V2 配置均启用 `train.ema_decay: 0.999`。每次成功更新优化器后，对动作头（含融合层）更新 EMA 权重；冻结 CLIP 不复制。设为 `0.0` 可关闭。checkpoint 同时保存原始权重 `head`、EMA 权重 `ema_head` 及 EMA 更新次数；断点恢复使用原始权重恢复训练，并恢复 EMA 状态。

　　启用 EMA 时，验证同时记录 `raw_val_loss` 和 `ema_val_loss`，其中 `val_loss` 采用 EMA 结果，`best.pt` 按最低 EMA 验证 loss 选择。`best.pt` 和 `latest.pt` 均含两套权重。评估与推理测速默认 `--weights auto`：有 EMA 时使用 EMA，旧 checkpoint 则使用原始权重；可指定 `--weights raw` 或 `--weights ema` 比较，实际使用的类型写入结果 JSON。

　　旧 checkpoint 未启用 EMA，直接 `--resume` 会保留原训练方式。若要从已有权重开始 EMA 微调，需使用开启 EMA 的配置和 `--init-checkpoint`，并指定新的输出目录；该操作重置优化器和 EMA 更新次数。有 EMA 的初始化 checkpoint 默认使用 EMA 权重作为微调起点。

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
  --checkpoint outputs/train/libero_spatial_v1/best.pt \
  --suite libero_spatial --episodes 10 --execute-steps 4 \
  --output-dir outputs/eval/libero_spatial_v1
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

### 4.3 推理耗时测试

​	使用 checkpoint 中的模型结构测试 batch=1 的完整动作块推理，无需启动仿真：

```bash
HF_HUB_OFFLINE=1 uv run scripts/measure_inference_time.py \
  --checkpoint outputs/train/libero_spatial_v1/latest.pt \
  --device cuda --precision fp32 \
  --output outputs/benchmark/libero_spatial_v1.json

HF_HUB_OFFLINE=1 uv run scripts/measure_inference_time.py \
  --checkpoint outputs/train/libero_spatial_v2/latest.pt \
  --device cuda --precision fp32 \
  --output outputs/benchmark/libero_spatial_v2.json
```

具体流程是：

1.  使用 **单张 GPU、batch=1**，加载 checkpoint，模型进入评估模式并关闭梯度，报告记录实际 GPU 型号。
2.  提前准备固定的合成图像、状态和分词后的语言输入，并放到 GPU。脚本使用 eager 模式，不启用编译或 CUDA Graph；输入准备和传输不计时。
3.  **预热 10 次**，每次推理后同步 GPU。
4.  正式测量 **50 次**：每次开始前同步 GPU，生成完整动作块后再次同步，再记录耗时。
5.  取 50 次耗时的**中位数**作为延迟，同时报告平均值、P95、最小值、最大值和各次耗时。动作长度从 checkpoint 读取；使用当前 LIBERO 配置新训练的模型时，每次生成 **12 步动作**，旧的 8 步权重仍生成 8 步。

​	计时包含视觉编码、语言编码、融合、动作头及 GPU 上的动作反归一化、夹爪转换和裁剪；不包含模型加载、输入准备和传输、输出拷回 CPU、通信或环境执行。为包含完整语言编码，脚本绕过正常评估的语言缓存，每次重新执行文本编码器。

​	默认使用 FP32；可通过 `--precision fp16` 或 `bf16` 测试混合精度，通过 `--warmup` 和 `--runs` 调整预热与测量次数。输出的每秒动作块预测次数由中位延迟换算，不代表机械臂实际控制频率。



| 项目           | 你的 V1 / V2                  | TurboVLA LIBERO                           |
| -------------- | ----------------------------- | ----------------------------------------- |
| 训练数据       | 仅 Spatial，10 个任务         | Spatial、Object、Goal、Long，共 40 个任务 |
| 数据处理       | 原始 HDF5，90% 训练、10% 验证 | 去除无效动作的 `no_noops` RLDS 数据       |
| 图像尺寸       | 224×224，双相机               | 256×256，双相机                           |
| 视觉编码器     | **CLIP ViT-B/16，冻结**       | **DINOv3 ViT-B，参与训练**                |
| 语言编码器     | CLIP 文本编码器，冻结         | BERT，冻结                                |
| 融合模块初始化 | 随机初始化；V1 无融合         | 从 **GroundingDINO** 加载相关预训练参数   |
| 融合层数       | V1：0；V2：6                  | 6                                         |
| 动作解码层数   | 4                             | 3                                         |
| 动作预测长度   | 12                            | 12                                        |
| 损失           | 前六维 Smooth L1 + 夹爪 BCE   | 七维动作的 masked L1                      |
| 动作归一化     | 前六维均值／标准差            | 前六维 min/max 映射至 `[-1,1]`            |
| 图像增强       | 平移 4 像素，颜色扰动 0.1     | 本地数据加载路径未见随机图像增强          |

优化器和训练计划也不同：

| 参数               | 你的 V1 / V2        | TurboVLA 官方方案                        |
| ------------------ | ------------------- | ---------------------------------------- |
| GPU 数量           | 单卡                | 4 卡                                     |
| 每卡 batch         | 16                  | 8                                        |
| 梯度累积           | 4                   | 4                                        |
| **全局有效 batch** | **64**              | **128**                                  |
| 总训练步数         | 20,000              | 40,000¹                                  |
| 峰值学习率         | `1e-4`              | `5e-5`                                   |
| Warmup             | 500 步              | 10,000 步                                |
| Warmup 后学习率    | 余弦下降至约 `1e-5` | 保持峰值                                 |
| Weight decay       | `1e-3`              | `1e-10`                                  |
| AdamW betas        | `(0.9, 0.999)`      | `(0.9, 0.95)`                            |
| EMA 参数滑动平均   | 无                  | 有，衰减系数 `0.999`                     |
| 计算精度           | FP16 autocast       | 策略参数 FP32，DINOv3 使用 BF16 autocast |
| 梯度裁剪           | 1.0                 | 1.0                                      |
| 训练 seed          | 42                  | 42                                       |
