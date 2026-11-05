# MetaWorld MT50 接入验证（2026-11-05）

已实现官方 LeRobot v3 MT50 图像/parquet 数据导入、共享行为克隆训练、MetaWorld 闭环评估。

## 自动检查

独立 `.venv-mt50` 环境中：

```bash
.venv-mt50/bin/pytest -q
.venv-mt50/bin/ruff check src tests
```

23 项测试通过，Ruff 通过。覆盖原有 LIBERO、旧默认模型兼容、新 MT50 数据跨文件轨迹、
按任务划分演示、仅训练集归一化、连续夹爪动作、动作裁剪、尾部 mask、原子导入、
含融合层的精确断点恢复，以及相机方向、动作队列清理和任意一步成功计数。

## 真实数据与 GPU 训练

从 `lerobot/metaworld_mt50` 下载了 meta/info.json、meta/tasks.parquet、
meta/episodes/chunk-000/file-000.parquet 和第一个数据分片 data/chunk-000/file-000.parquet
（99,730,976 字节），未下载完整数据集。

通过 `--max-episodes 2` 导入前两条 assembly-v3 演示，共 179 帧：训练 88 帧、验证 91 帧。
90 帧专家动作超出 [-1,1]，已裁剪为有效控制范围，详见：
`data/metaworld_mt50_probe/conversion.json`。

配置基于 configs/metaworld_mt50.yaml，探测时设置 batch=2、累积=1、steps=10，
每 5 步验证和保存。使用本地真实 CLIP、CUDA/FP16，10 步训练完成，所有 loss/梯度有限。
第 10 步固定小验证集 loss=0.387823，峰值 PyTorch 张量显存=0.720019 GiB。
可训练参数 7,773,964；该显存不包括全部 CUDA/EGL/桌面占用。

权重和训练日志：`runs/mt50_integration/train/`。
仅两条演示、10 步训练用于验证接口，不代表 MT50 训练效果或推荐权重。

## 真实模拟器

固定依赖：MetaWorld 3.0.0、MuJoCo 3.3.0、Gymnasium 1.1.1。
Gymnasium 1.1.0 存在 render_modes 断言不兼容，已通过 1.1.1 固定版本修复，未改第三方源码。

- `eval-metaworld --protocol lerobot`：assembly-v3，2 回合 × 20 步，GPU/EGL 渲染、模型控制、
  JSON 结果与两个 RGB MP4 保存成功。10 步探测权重的成功数为 0/2，属于预期的接口检查。
- `eval-metaworld --protocol benchmark`：实际构造 MT50 benchmark，assembly-v3，1 回合 × 5 步。
  正常输出 goal_index=5、goal_sha256 与结果，验证了另一条协议分支。
- `reach-v3` 原生专家：第 41 步触发 info.success。RGB 为 480×480×3 uint8。
  同一环境重复显式 env.seed(123) 后 reset，完整状态逐元素相同。

实际结果：

- runs/mt50_integration/runtime_check.json
- runs/mt50_integration/eval/results.json
- runs/mt50_integration/eval_benchmark/results.json
- runs/mt50_integration/reach_rgb.png

## 尚未完成的实验

没有下载或训练全量 50 任务，没有测量完整 MT50 成功率。
完整原始 parquet 当前合计约 45.9GB，还需要转换后 HDF5 的磁盘空间。
完整训练和评估的使用方法见 README 的 MetaWorld MT50 一节。
