# myVLA 模型历史

## 一、VLA框架：

- 输入：224×224 双相机 RGB 图像、任务语言、8 维机器人状态。
- 视觉编码器：预训练 CLIP ViT-B/16，V1/V1.1/V1.2 冻结，V2/V2.1/V2.2/V2.3 参与训练；V4 换用 SigLIP，详见下文。
- 文本编码器：CLIP 文本编码器，所有版本均保持冻结。
- 输出：当前各版本训练配置均为 `chunk_size: 12`，输出 12×7 动作块。
- 损失函数：前 6 维 Smooth L1，夹爪 BCE；排除轨迹尾部 padding。

## 二、V1、V2六个版本对比

​	V1 系列（V1、V1.1、V1.2）冻结 CLIP 视觉编码器；V2 系列（V2、V2.1、V2.2）训练视觉编码器。每个系列分别提供 0、2、6 层双向视觉—语言交叉注意力融合块。文本编码器在六个版本中均冻结。

### 模型结构与训练范围

| 版本 | 交叉注意力融合块 | CLIP 视觉编码器 | 融合层/动作头学习率 | 视觉编码器学习率 |
|---|---:|---|---:|---:|
| V1 | 0 层 | 冻结 | `1e-4` | — |
| V1.1 | 2 层 | 冻结 | `1e-4` | — |
| V1.2 | 6 层 | 冻结 | `1e-4` | — |
| V2 | 0 层 | 训练 | `1e-4` | `1e-5` |
| V2.1 | 2 层 | 训练 | `1e-4` | `1e-5` |
| V2.2 | 6 层 | 训练 | `1e-4` | `1e-5` |

​	六个版本均使用 4 层动作解码器，预测 12×7 动作块。表中融合块位于动作解码器之前，融合块层数不包含动作解码器内部的注意力层。无额外融合块的版本仍通过动作解码器读取多模态 memory。

​	训练范围还包括视觉/语言特征投影、状态 MLP、动作查询及相机/模态嵌入；CLIP 自带的原始投影层和温度参数均冻结，前向未使用。

### 参数规模

| 版本 | 总参数量（M） | 可训练参数量（M） | 冻结参数量（M） | 一套 FP32 完整权重 |
|---|---:|---:|---:|---:|
| V1 | 154.24 | 4.62 | 149.62 | 588.37 MiB |
| V1.1 | 157.40 | 7.78 | 149.62 | 600.42 MiB |
| V1.2 | 163.72 | 14.10 | 149.62 | 624.53 MiB |
| V2 | 154.24 | 90.42 | 63.82 | 588.37 MiB |
| V2.1 | 157.40 | 93.58 | 63.82 | 600.42 MiB |
| V2.2 | 163.72 | 99.89 | 63.82 | 624.53 MiB |

​	相同融合深度的两组版本（V1/V2、V1.1/V2.1、V1.2/V2.2）结构与总参数量相同，区别在于视觉编码器是否更新。FP32 大小仅指一套完整模型参数，不等于包含优化器、EMA 等状态的 checkpoint 大小。

### 配置与训练目录

​	六个版本的默认训练设置一致：使用全部训练数据、不划分验证集；batch=16、梯度累积=4（有效 batch=64），训练 10,000 步，500 步 warmup，权重衰减 `0.001`，EMA 衰减 `0.999`，每 2,000 步保存 checkpoint。训练期间不计算验证 loss，不启动定期仿真评估。历史训练的实际设置以对应 checkpoint 中的配置为准。

## 三、验证成功率：

### 单套件验证成功率：

​	以下记录各次评估在 **LIBERO-Spatial** 上的闭环成功率。列头为 checkpoint 的实际训练步数；“—”表示暂无评估结果。模型版本依据评估配置中的架构与 checkpoint 路径判定，不仅依据输出目录名。

| 模型 | 2,000 步 | 4,000 步 | 6,000 步 | 8,000 步 | 10,000 步 |
|---|---:|---:|---:|---:|---:|
| V1 | 2% | 76% | 89% | 81% | 84% |
| V1.1 | 13% | 78% | 80% | 83% | 84% |
| V1.2 | 19% | 74% | 80% | 82% | 79% |
| V2 | 26% | 91% | 89% | 87% | 92% |
| V2.1 | 18% | 70% | 88% | 86% | 85% |
| V2.2 | 26% | 86% | 95% | 91% | 96% |
| V2.3 | 25% | 86% | 90% | 90% | 88% |

### V2.2 四套件联合训练评估：

​	本组使用 6 层双向视觉—语言融合的 V2.2，视觉编码器参与训练，文本编码器冻结。训练覆盖 LIBERO-Spatial、LIBERO-Object、LIBERO-Goal 和 LIBERO-10。评估使用 EMA 权重、`seed=42`；每套件 10 个任务，每任务 10 次，共 400 次评估。各 checkpoint 均已完成四套件评估。

| 训练步数 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | 总成功次数 | 总体成功率 |
|---|---:|---:|---:|---:|---:|---:|
| 8,000 | 92% | 100% | 96% | 80% | 368/400 | 92.00% |
| 16,000 | 91% | 95% | 95% | 88% | 369/400 | 92.25% |
| 24,000 | 96% | 99% | 98% | 92% | 385/400 | **96.25%** |
| 32,000 | 95% | 98% | 95% | 88% | 376/400 | 94.00% |
| 40,000 | 94% | 94% | 96% | 87% | 371/400 | 92.75% |

​	当前 24,000 步的总体成功率最高。16,000 步相比 8,000 步，LIBERO-10 提升 8 个百分点，其他三个套件有所下降，总体多成功 1 次。32,000 和 40,000 步的总体成功率分别比 24,000 步下降 2.25 和 3.50 个百分点，继续训练未带来更高的评估成功率。上述结果为单个评估种子的观测结果。

​	原始记录位于 `outputs/eval/libero_all_v2_2/ckpt{8000,16000,24000,32000,40000}/seed42/<suite>/`，成功次数已与逐回合记录及 `log.json` 核对。本组与上表的 Spatial 单套件训练实验分别统计；此前缺少 LIBERO-10 训练数据的三套件实验也不计入本组。

## 四、V3—Flow Matching 动作头

​	V3 基于 V2.2，保留可训练 CLIP 视觉编码器、冻结文本编码器、6 层双向视觉—语言融合以及 4 层 Transformer 动作解码器。区别在于解码器接收带噪动作和时间嵌入，输出动作速度场，替代直接回归动作。

- 配置：`model.action_head: flow_matching`，`model.flow_steps: 10`；旧配置默认 `regression`，旧 checkpoint 仍可加载。
- 训练：采样独立高斯噪声 `x0` 和均匀时间 `t ∈ [0,1)`，构造 `xt=(1-t)x0+t*a`，用有效时间步上的 MSE 回归速度 `a-x0`。每个动作块共享一个时间，尾部 padding 的目标和噪声置零且不计入损失。
- 动作表示：LIBERO 前 6 维继续使用原有标准化；夹爪由数据层的 `{0,1}` 转为 `{-1,+1}`，与其他维度一起学习速度场，不再使用 BCE。推理时按 0 阈值转换夹爪，复用现有动作解码接口。MetaWorld 的 4 维动作均保持原有标准化。
- 推理：从高斯噪声出发，在 `t=0→1` 上执行 `flow_steps` 次等步长 Euler 更新；视觉编码和融合每次动作块预测只计算一次。默认输出仍为 `12×7`。采样使用 PyTorch RNG，同种子和相同调用顺序可复现。
- 配置分别为 `configs/libero_spatial/train_cfg_v3.yaml` 和 `configs/libero_all/train_cfg_v3.yaml`，训练设置分别沿用对应 V2.2，输出到独立的 `*_v3` 目录。

```bash
# 四套件联合训练（Spatial 单套件使用 configs/libero_spatial/train_cfg_v3.yaml）
unset UV_PROJECT_ENVIRONMENT
HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --config configs/libero_all/train_cfg_v3.yaml

# 闭环评估；动作头类型和积分步数由 checkpoint 配置自动恢复
UV_PROJECT_ENVIRONMENT=.venv-libero HF_HUB_OFFLINE=1 MUJOCO_GL=egl \
uv run --extra libero src/myvla/evaluate.py \
  --checkpoint outputs/train/libero_all_v3/checkpoint_40000.pt \
  --suite libero_all --episodes 10 --execute-steps 12 \
  --output-dir outputs/eval/libero_all_v3
```

​	V3 需要独立训练，不能用 V2.2 checkpoint 直接续训或通过 `--init-checkpoint` 跨动作头初始化。EMA、断点恢复和推理耗时脚本沿用现有流程；V3 的耗时包含完整的多步动作生成。当前尚无 V3 的真实训练或仿真成功率结果，损失数值也不能直接与 V2.2 的 Smooth L1+BCE 对比。

## 末、结论

### 数据集验证集划分：

​	一开始我是将训练数据集分为90%训练和10%固定验证集，在800固定验证帧上计算loss来作为验证集结果。但是发现按照验证loss选择出的best.pt权重在任务完成上成功率还不如继续训练的。

​	后来我直接不划分训练集和验证集，全部数据用于训练，发现任务成功率会替身大约5%百分点。

### 输出步长：

​	需要注意训练的时候要按什么action_chunk大小来训练，评估的时候如果action_chunk小于训练的action_chunk，会截取训练action_chunk的前面部分来作为动作。

## V4：SigLIP 编码器 + V2.2 回归动作头

V4 将 V2.2 的 CLIP 视觉和文本编码器替换为 `google/siglip-base-patch16-224`，保留 6 层双向融合、4 层动作解码器、12 步动作块及原有 Smooth L1 + 夹爪 BCE 损失。视觉编码器继续训练（`1e-5`），文本编码器冻结，融合和动作头学习率为 `1e-4`；单套件及联合训练配置分别沿用对应 V2.2 的训练参数。

适配内容包括：保留 SigLIP 全部 196 个视觉 patch（没有 CLS），文本固定 padding/truncation 到 64 token，不对固定 padding 加 attention mask，使用模型原生文本 pooled output。融合摘要继续使用最后一个 token 的交互残差；SigLIP 的该 token 是固定序列末尾 token。数据与增强接口沿用 CLIP 图像规范，编码器内部先还原 RGB，再转成 SigLIP 的 mean/std=0.5 规范，保证训练、闭环评估及耗时测量一致。

V4 配置采用固定上游 revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`，使用本地 `models/siglip-base-patch16-224`。需要另外下载 SigLIP 预训练权重（约 813 MB）；原 CLIP 权重不能替代它。新增依赖 `sentencepiece` 和 `protobuf` 已写入 `pyproject.toml` 和 `uv.lock`。

```bash
# 通用训练环境安装/更新依赖
unset UV_PROJECT_ENVIRONMENT
uv sync --locked

# 下载固定版本的 SigLIP 权重及 tokenizer；支持 --endpoint 指定镜像
uv run scripts/download_siglip.py --workers 2

# 四套件联合训练；Spatial 使用 configs/libero_spatial/train_cfg_v4.yaml
HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --config configs/libero_all/train_cfg_v4.yaml

UV_PROJECT_ENVIRONMENT=.venv-libero HF_HUB_OFFLINE=1 MUJOCO_GL=egl \
uv run --extra libero src/myvla/evaluate.py \
  --checkpoint outputs/train/libero_all_v4/checkpoint_40000.pt \
  --suite libero_all --episodes 10 --execute-steps 12 \
  --output-dir outputs/eval/libero_all_v4
```

默认输出分别为 `outputs/train/libero_spatial_v4` 和 `outputs/train/libero_all_v4`。MetaWorld 训练可用同一配置加 `--dataset metaworld`，仍使用 `.venv`；LIBERO 评估使用 `.venv-libero` 和 `--extra libero`，MT50 评估使用 `.venv-mt50` 和 `--extra mt50`。V4 需要独立训练，不能直接加载 V2.2 或 V3 的任务 checkpoint。旧 checkpoint 仍按各自保存的 backend 和结构恢复；早期曾使用 V4 名称的 8 层 CLIP 融合实验归入 V2.3，新 V4 专指本节的 SigLIP 模型。

当前只完成离线小模型功能验证，尚未下载完整 SigLIP 权重或开展正式训练、GPU 显存测量与仿真成功率评估。

参考：[SigLIP 模型与 tokenizer 文档](https://huggingface.co/docs/transformers/model_doc/siglip)、[预训练权重](https://huggingface.co/google/siglip-base-patch16-224)。


## V5：SigLIP 编码器 + Flow Matching 动作头

V5 组合 V4 的 SigLIP Base patch16-224 视觉/文本编码器和 V3 的条件 Flow Matching 动作头。沿用 6 层双向视觉—语言融合、4 层 Transformer 动作解码器和 12 步动作块；视觉编码器参与训练（学习率 `1e-5`），文本编码器冻结，融合模块和动作头使用 `1e-4`。

| 模型 | 编码器 | 动作头 |
|---|---|---|
| V2.2 | CLIP | 直接回归 |
| V3 | CLIP | Flow Matching |
| V4 | SigLIP | 直接回归 |
| V5 | SigLIP | Flow Matching |

V5 的 `model.backend: siglip`、`model.action_head: flow_matching`、`model.flow_steps: 10`。训练使用 V3 的带 mask 速度 MSE，LIBERO 夹爪转换为有符号动作；推理默认从高斯噪声开始执行 10 步 Euler 积分。图像归一化、196 个视觉 patch 及 64 token 固定文本处理均沿用 V4。LIBERO 输出为 `12×7`；切换 MT50 后为单相机、4 维状态、`12×4` 动作块。

配置分别为 `configs/libero_spatial/train_cfg_v5.yaml` 和 `configs/libero_all/train_cfg_v5.yaml`，训练超参数分别沿用对应的 V3/V4，输出到独立的 `*_v5` 目录。已有 V4 的 `models/siglip-base-patch16-224` 权重可复用，不需要另下载 Flow Matching 权重，也没有新增软件依赖。

```bash
# 所有训练使用 .venv；如缺少 SigLIP 权重，先运行 scripts/download_siglip.py
unset UV_PROJECT_ENVIRONMENT
HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --config configs/libero_all/train_cfg_v5.yaml

# LIBERO 四套件评估
UV_PROJECT_ENVIRONMENT=.venv-libero HF_HUB_OFFLINE=1 MUJOCO_GL=egl \
uv run --extra libero src/myvla/evaluate.py \
  --checkpoint outputs/train/libero_all_v5/checkpoint_40000.pt \
  --suite libero_all --episodes 10 --execute-steps 12 \
  --output-dir outputs/eval/libero_all_v5

# MT50 训练，使用独立的 MT50 演示数据
unset UV_PROJECT_ENVIRONMENT
HF_HUB_OFFLINE=1 uv run src/myvla/train.py \
  --config configs/libero_spatial/train_cfg_v5.yaml --dataset metaworld

# MT50 评估
UV_PROJECT_ENVIRONMENT=.venv-mt50 HF_HUB_OFFLINE=1 MUJOCO_GL=egl \
uv run --extra mt50 src/myvla/evaluate.py \
  --checkpoint outputs/train/metaworld_mt50_v5/checkpoint_10000.pt \
  --episodes 50 --execute-steps 1 --protocol lerobot \
  --output-dir outputs/eval/metaworld_mt50_v5
```

V5 需要独立训练，现有 V3/V4 的任务 checkpoint 不能直接跨版本续训。EMA、断点恢复和推理耗时测量复用现有流程。当前完成的是离线小模型训练、梯度、EMA、断点恢复及推理一致性验证，尚无 V5 正式训练或仿真成功率结果。
