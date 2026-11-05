# Spatial 优化实验记录

日期：2026-11-05（Asia/Shanghai）。

## 结果

| 模型 | 成功回合 / 100 | 完整验证集 loss |
|---|---:|---:|
| 原 best（9,000 步） | 67 | 0.277981 |
| 原 latest（30,000 步） | 70 | 0.328802 |
| 增强微调 best（额外 2,500 步） | 82 | 0.261862 |

相对原 best：同一初始状态上，25 个回合由失败变为成功，10 个由成功变为失败。

## 每个任务的成功次数（各 10 回合）

| 任务 ID | 原 best | 原 latest | 增强微调 |
|---|---:|---:|---:|
| 0 | 7 | 7 | 10 |
| 1 | 6 | 1 | 9 |
| 2 | 10 | 10 | 8 |
| 3 | 6 | 8 | 7 |
| 4 | 8 | 4 | 7 |
| 5 | 6 | 8 | 5 |
| 6 | 4 | 10 | 10 |
| 7 | 9 | 10 | 10 |
| 8 | 6 | 8 | 10 |
| 9 | 5 | 4 | 6 |

## 改动与验证

- 保持冻结 CLIP 和 4 层、256 维动作头结构；从原 best.pt 初始化。
- 仅在训练时执行最多 4 像素平移和 0.1 强度的亮度/对比度增强，不做翻转或旋转。
- 学习率 3e-5、权重衰减 1e-3、warmup 100 步，总计微调 3,000 步。
- micro batch=4、梯度累积=4，有效 batch 仍为 16。
- 增强在主训练进程内使用 torch RNG；包含增强的 CPU 断点恢复逐位一致性测试通过。
- 新增 --init-checkpoint，仅加载权重，重置优化器和学习率计划；检查架构、数据划分和归一化一致性。
- 13 项测试通过，Ruff 静态检查通过。
- 训练中的固定验证样本仍为 200 帧；另对三份模型统一评估完整 6,224 帧验证集。

## 评测协议与边界

LIBERO-Spatial，10 任务 × 10 初始状态，seed=42，horizon=220，settle_steps=10，execute_steps=4。初始状态索引及文件校验和一致。

这是同一开发评测集上的比较，尚未以独立 seed 确认提升，也未验证其他 LIBERO 套件；不能把单任务 10/10 当作稳定 100% 成功率。

## 产物

- 推荐候选权重：runs/spatial_aug_v2/best.pt
- 微调配置：configs/libero_spatial_aug.yaml
- 微调曲线：runs/spatial_aug_v2/loss_curve.png
- 新模型仿真结果：runs/optimize/aug_full/results.json
- 完整验证集对照：runs/optimize/full_validation.json
- 机器可读对照报告：runs/optimize/comparison.json

复现时使用新输出目录：

```bash
HF_HUB_OFFLINE=1 uv run --locked myvla train \
  --config configs/libero_spatial_aug.yaml \
  --init-checkpoint runs/spatial_v1/best.pt \
  --output-dir runs/spatial_aug_repeat
```
