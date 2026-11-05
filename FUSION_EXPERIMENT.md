# 双向视觉—语言融合实验（2026-11-05）

本轮融合模型在相同 100 回合中成功 84 次；原模型 82 次、相同额外训练量的无融合对照 83 次。
差距很小，尚不足以确认架构带来了稳定收益。保留原模型，融合版本作为候选实验权重。

## 改动

保留冻结的 CLIP ViT-B/16，使用逐 token 文本输出和 padding mask。
新增 2 层、256 维、8 头的双向交叉注意力，每个方向含残差 FFN。
四个 tanh 残差门控从零初始化，使加载旧权重后的 eval 输出与原模型一致。
视觉特征与文本 token 交互；最终文本 EOS 的交互增量加回原 pooled 文本特征，
再与视觉、状态一起送入原动作解码器。

总参数 157,396,752（约 0.157B），可训练参数 7,776,015；新增 3,159,048 参数。
这不是 TurboVLA 完整复现：仍使用 CLIP，没有使用 DINOv3、BERT 或 grounding 预训练融合权重。

`model.fusion_layers=0` 为兼容旧模型的默认值。`--init-checkpoint` 仅允许原架构扩展融合层，
其余结构、数据划分和归一化必须一致；新增参数只允许来自 fusion 模块。
`--resume` 仍要求架构一致。

## 训练与验证

两组均从 spatial_aug_v2/best.pt 初始化，额外训练 3,000 步：batch=4、累积=4、
lr=3e-5、warmup=100、weight_decay=1e-3、image_shift=4、color_jitter=0.1。
帧采样与数据划分相同；新增模块会改变随机数消耗，因此 dropout/增强的逐次抽样并非完全配对。
按固定 200 帧验证 loss 选择最佳 checkpoint，融合选中第 3,000 步，对照选中第 2,500 步。

| 模型 | 完整验证集 loss（6,224 帧） | 仿真成功率 | 训练峰值张量显存 |
|---|---:|---:|---:|
| 原增强模型 | 0.261862 | 82/100 | 约 0.696 GiB（此前记录） |
| 原架构额外训练 | 0.264945 | 83/100 | 0.696 GiB |
| 双向融合 | 0.260624 | 84/100 | 0.774 GiB |

显存为 PyTorch max_memory_allocated，不含全部 CUDA/EGL/桌面开销。
两次训练并行执行，用时约 817 秒（对照）、882 秒（融合），不作为独占 GPU 性能基准。
融合门控最终绝对值均小于 0.0002，交互更新幅度较小。
本轮采用保守短微调，不能据此否定更充分训练或预训练融合模块的价值。

## 闭环评估

LIBERO-Spatial，seed=42，10 任务 × 10 回合，horizon=220，settle_steps=10，execute_steps=4。
三组协议、初始状态索引及状态文件 SHA256 已核对一致。

| 任务 ID | 原模型 | 继续训练对照 | 融合 |
|---|---:|---:|---:|
| 0 | 10/10 | 10/10 | 10/10 |
| 1 | 9/10 | 5/10 | 6/10 |
| 2 | 8/10 | 9/10 | 9/10 |
| 3 | 7/10 | 8/10 | 9/10 |
| 4 | 7/10 | 9/10 | 9/10 |
| 5 | 5/10 | 5/10 | 5/10 |
| 6 | 10/10 | 10/10 | 9/10 |
| 7 | 10/10 | 10/10 | 10/10 |
| 8 | 10/10 | 10/10 | 9/10 |
| 9 | 6/10 | 7/10 | 8/10 |

融合相对原模型：9 回合由失败变成功，7 回合由成功变失败。
融合相对继续训练对照：7 回合由失败变成功，6 回合由成功变失败。
任务 1 从原来的 9/10 降到 6/10，存在局部退步。
这是同一开发评测集的单次训练结果，未做多训练种子、独立评估种子或关闭融合门控的消融。
不能将净增 1–2 个回合解释为统计可靠的架构提升。

## 复现与产物

```bash
HF_HUB_OFFLINE=1 uv run --locked myvla train \
  --config configs/libero_spatial_fusion.yaml \
  --init-checkpoint runs/spatial_aug_v2/best.pt \
  --output-dir runs/spatial_fusion_repeat

HF_HUB_OFFLINE=1 MUJOCO_GL=egl uv run --locked --extra sim myvla eval \
  --checkpoint runs/spatial_fusion_v3/best.pt \
  --episodes 10 --output runs/optimize/fusion_repeat_eval \
  --libero-root /home/lpx/python-project/LIBERO
```

- 候选模型：runs/spatial_fusion_v3/best.pt
- 原模型：runs/spatial_aug_v2/best.pt
- 对照配置：configs/libero_spatial_fusion_control.yaml
- 曲线：runs/spatial_fusion_v3/loss_comparison.png
- 完整验证：runs/optimize/fusion_validation.json
- 仿真：runs/optimize/fusion_full/results.json、runs/optimize/fusion_control_full/results.json
- 汇总：runs/optimize/fusion_comparison.json

17 项 pytest 测试及 Ruff 检查通过，覆盖门控初始输出一致、padding 不影响有效 token、
双向注意力梯度、冻结编码器、跨架构初始化、checkpoint 加载，以及含融合/增强的精确断点恢复。
