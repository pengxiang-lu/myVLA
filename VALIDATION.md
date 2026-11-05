# 本机验证记录

日期：2026-11-01（Asia/Shanghai）。代码、uv.lock 与 `.venv` 均位于
`/home/lpx/python-project/myVLA/`。

## 环境

- uv 管理的 Python 3.10.20，PyTorch 2.6.0+cu124，Transformers 4.57.6。
- RTX 3060 Laptop GPU，`nvidia-smi` 显示 6144MiB 显存；驱动 595.99.02。
- 在实际执行环境中 CUDA 可用；未修改系统驱动。
- LIBERO 读取现有源码 `/home/lpx/python-project/LIBERO`，不修改其文件。
- 此源码目录没有 Git 元数据；评测会额外记录所用初始状态文件的 SHA256。

## 已通过

1. `uv sync --locked --extra sim`：独立环境安装成功。
2. `uv run --locked --extra sim ruff check src tests scripts`：静态检查通过。
3. `uv run --locked --extra sim pytest -q`：6 项测试通过。
   - 完整轨迹的训练/验证隔离、训练集统计量与尾部动作 mask。
   - mask 排除 padding 的损失与梯度。
   - CLIP 始终冻结、动作头确实更新。
   - 动作转换、夹爪符号、四元数转换与图像方向。
   - 中断恢复后的参数与连续训练逐位一致（CPU）。
   - 控制器重规划、回合之间清空队列与成功判定。
4. `uv run --no-sync myvla smoke --output runs/smoke`：3 次训练更新、离线验证、
   checkpoint 保存/重载、输出 `(1, 8, 7)` 均通过。
5. `uv run --no-sync python scripts/profile_cuda.py`：随机初始化的完整尺寸
   CLIP ViT-B/16 + 4 层 256 维动作头，完成 3 次 CUDA/FP16 前后向。
   双相机 224×224，micro batch=2，可训练参数 4,616,967，
   PyTorch 峰值分配 0.6641GiB，loss 有限、骨干没有梯度。
6. `uv run --no-sync myvla sim-smoke --libero-root /home/lpx/python-project/LIBERO --gl egl`：
   真实 LIBERO Spatial task 0 的 reset、官方初始状态恢复及 5 次 step 通过；
   两个相机均为 `(128,128,3)`，状态为 8 维。

## 边界与已知问题

- CPU smoke 使用合成 HDF5 和随机微型 CLIP；CUDA profile 使用完整尺寸随机 CLIP。
  两者都不验证预训练特征质量，也不是 LIBERO 成功率。
- 尚未下载 CLIP 预训练权重或 LIBERO 演示数据，未启动正式 LIBERO 训练，
  尚无训练后策略的真实仿真成功率。
- OSMesa 在本机发生段错误；EGL 已通过。实际训练后的评测使用 EGL。
- 系统 ROS 通过 PYTHONPATH 注入 pytest 插件，最初造成测试启动失败；
  已通过 pyproject.toml 的 `--disable-plugin-autoload` 隔离。
- LIBERO/BDDL 未声明的运行时依赖（matplotlib、future 等）已显式纳入 sim extra。
