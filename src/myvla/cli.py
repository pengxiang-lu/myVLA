import argparse
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from .config import Config, DataConfig, ModelConfig, TrainConfig
from .data import compute_stats, manifest, scan_episodes
from .train import add_train_arguments, policy_from_checkpoint, run_training, train


def doctor():
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "libero_installed": importlib.util.find_spec("libero") is not None,
    }
    if torch.cuda.is_available():
        report["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_vram_gib": torch.cuda.get_device_properties(i).total_memory / 2**30,
            }
            for i in range(torch.cuda.device_count())
        ]
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        report["nvidia_smi"] = (result.stdout + result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        report["nvidia_smi"] = str(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def make_smoke_data(directory):
    """Synthetic LIBERO-shaped data, intentionally not a benchmark result."""
    import h5py

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "smoke_demo.hdf5"
    if path.exists():
        return path
    rng = np.random.default_rng(7)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "pick up the bowl"})
        for i in range(4):
            ep = data.create_group(f"demo_{i}")
            n = 5 + i
            actions = rng.uniform(-0.5, 0.5, (n, 7)).astype(np.float32)
            actions[:, 6] = rng.choice([-1, 1], size=n)
            ep.create_dataset("actions", data=actions)
            obs = ep.create_group("obs")
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                obs.create_dataset(key, data=rng.integers(0, 256, (n, 32, 32, 3), np.uint8))
            obs.create_dataset("ee_states", data=rng.normal(size=(n, 6)).astype(np.float32))
            obs.create_dataset("gripper_states", data=rng.normal(size=(n, 2)).astype(np.float32))
    return path


def smoke(output):
    torch.set_num_threads(2)
    out = Path(output)
    make_smoke_data(out / "synthetic_data")
    config = Config(
        ModelConfig(
            backend="tiny",
            image_size=32,
            hidden_dim=32,
            layers=1,
            heads=4,
            chunk_size=8,
            dropout=0.1,
        ),
        DataConfig(directory=str(out / "synthetic_data"), val_fraction=0.25),
        TrainConfig(
            output_dir=str(out / "training"),
            device="cpu",
            precision="fp32",
            batch_size=2,
            grad_accumulation=2,
            workers=0,
            steps=3,
            warmup_steps=1,
            validate_every=2,
            validation_batches=2,
            save_every=1,
            log_every=1,
        ),
    )
    checkpoint = train(config)
    policy, _, _, payload = policy_from_checkpoint(checkpoint, torch.device("cpu"))
    with torch.no_grad():
        result = policy(torch.zeros(1, 2, 3, 32, 32), torch.zeros(1, 8), ["pick up the bowl"])
    assert result.shape == (1, 8, 7) and torch.isfinite(result).all()
    print(
        json.dumps(
            {
                "smoke_test": "passed",
                "step": payload["step"],
                "prediction_shape": list(result.shape),
                "checkpoint": str(checkpoint),
                "note": "Synthetic data/random tiny CLIP; not a LIBERO success rate.",
            }
        )
    )


def sim_smoke(libero_root, gl):
    import os

    from .libero_runtime import load_init_states, setup_libero
    from .preprocess import state_from_observation

    os.environ["MUJOCO_GL"] = gl
    info = setup_libero(libero_root)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    states, digest = load_init_states(suite, 0)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(0), camera_heights=128, camera_widths=128
    )
    try:
        env.seed(42)
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(5):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        state = state_from_observation(obs)
        assert state.shape == (8,) and np.isfinite(state).all()
        images = {
            key: list(obs[key].shape) for key in ("agentview_image", "robot0_eye_in_hand_image")
        }
        assert all(shape == [128, 128, 3] for shape in images.values())
        print(
            json.dumps(
                {
                    "sim_smoke": "passed",
                    "images": images,
                    "state_dim": len(state),
                    "init_states_sha256": digest,
                    "libero": info,
                }
            )
        )
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="myVLA: frozen CLIP + action-chunk Transformer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect Python/PyTorch/CUDA without changing drivers")
    p = commands.add_parser(
        "sim-smoke", help="Reset/step a real LIBERO environment without a model"
    )
    p.add_argument("--libero-root")
    p.add_argument("--gl", choices=["egl", "osmesa"], default="egl")
    p = commands.add_parser("smoke", help="Offline CPU end-to-end check with synthetic HDF5")
    p.add_argument("--output", default="outputs/train/smoke")
    p = commands.add_parser("inspect", help="Validate LIBERO or converted MT50 HDF5 and split")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--pattern", default="*_demo.hdf5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", choices=["libero", "metaworld"], default="libero")
    p = commands.add_parser(
        "convert-metaworld", help="Import official LeRobot MT50 parquet to HDF5"
    )
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", type=int, nargs="+")
    p.add_argument("--max-episodes", type=int)
    p.add_argument("--image-size", type=int, default=224)
    p = commands.add_parser("eval-metaworld", help="Closed-loop MetaWorld MT50 evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tasks", type=int, nargs="+")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--protocol", choices=["lerobot", "benchmark"], default="lerobot")
    p.add_argument("--videos", action="store_true")
    p = commands.add_parser("train")
    add_train_arguments(p)
    p = commands.add_parser("eval", help="Closed-loop LIBERO simulator evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    p.add_argument("--tasks", type=int, nargs="+")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=int)
    p.add_argument("--videos", action="store_true")
    p.add_argument("--libero-root")
    p = commands.add_parser("download", help="Download LIBERO or raw MT50 from official HF source")
    p.add_argument(
        "--suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_100", "metaworld_mt50"],
    )
    p.add_argument("--directory", default="data")
    p.add_argument("--libero-root")
    p.add_argument("--endpoint", help="Hugging Face endpoint or compatible mirror URL")
    p.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        doctor()
    elif args.command == "sim-smoke":
        sim_smoke(args.libero_root, args.gl)
    elif args.command == "smoke":
        smoke(args.output)
    elif args.command == "inspect":
        episodes = scan_episodes(
            DataConfig(
                directory=args.data_dir,
                pattern=args.pattern,
                dataset=args.dataset,
                flip_vertical=args.dataset == "libero",
            ),
            args.seed,
        )
        stats = compute_stats(episodes, args.dataset)
        print(
            json.dumps(
                {
                    "episodes": manifest(episodes),
                    "stats": {k: v.tolist() for k, v in stats.items()},
                },
                indent=2,
            )
        )
    elif args.command == "convert-metaworld":
        from .metaworld_data import convert_metaworld

        report = convert_metaworld(
            args.source, args.output, args.tasks, args.max_episodes, args.image_size
        )
        print(json.dumps(report, indent=2))
    elif args.command == "eval-metaworld":
        from .metaworld_evaluate import evaluate_metaworld

        evaluate_metaworld(
            args.checkpoint,
            args.output_dir,
            args.tasks,
            args.episodes,
            args.device,
            args.execute_steps,
            args.seed,
            args.horizon,
            args.protocol,
            args.videos,
        )
    elif args.command == "train":
        run_training(args, parser, trainer=train)
    elif args.command == "eval":
        from .evaluate import evaluate

        evaluate(
            args.checkpoint,
            args.suite,
            args.tasks,
            args.episodes,
            args.device,
            args.execute_steps,
            args.output_dir,
            args.seed,
            args.horizon,
            args.videos,
            args.libero_root,
        )
    elif args.command == "download":
        from huggingface_hub import snapshot_download

        if args.max_workers < 1:
            parser.error("--max-workers must be positive")
        if args.suite == "metaworld_mt50":
            snapshot_download(
                repo_id="lerobot/metaworld_mt50",
                repo_type="dataset",
                local_dir=str((Path(args.directory) / "metaworld_mt50_raw").resolve()),
                allow_patterns=["meta/**", "data/**"],
                force_download=False,
                max_workers=args.max_workers,
                endpoint=args.endpoint,
            )
            return
        # Download directly: upstream deletes existing data and forces re-downloads.
        # Keep local_dir metadata and partial files so interrupted runs can resume.
        Path(args.directory).mkdir(parents=True, exist_ok=True)
        patterns = (
            ["libero_90/*.hdf5", "libero_10/*.hdf5"]
            if args.suite == "libero_100"
            else [f"{args.suite}/*.hdf5"]
        )
        print(
            "Downloading LIBERO; existing files and partial downloads will be retained.",
            flush=True,
        )
        snapshot_download(
            repo_id="yifengzhu-hf/LIBERO-datasets",
            repo_type="dataset",
            local_dir=str(Path(args.directory).resolve()),
            allow_patterns=patterns,
            force_download=False,
            max_workers=args.max_workers,
            endpoint=args.endpoint,
        )


if __name__ == "__main__":
    main()
