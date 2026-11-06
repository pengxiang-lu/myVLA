import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Support both package imports and `uv run src/myvla/train.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "myvla"

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from .config import from_dict, load_config
from .data import RobotDataset, compute_stats, manifest, scan_episodes
from .model import Policy, action_loss
from .preprocess import augment_images


def plot_loss(directory):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    directory = Path(directory)
    rows = []
    for line in (directory / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("No training metrics to plot")
    fig = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    train = [r for r in rows if "loss" in r]
    val = [r for r in rows if "val_loss" in r]
    ax.plot([r["step"] for r in train], [r["loss"] for r in train],
            color="#9bbbd5", alpha=0.6, linewidth=0.8, label="Train loss (logged batches)")
    if val:
        ax.plot([r["step"] for r in val], [r["val_loss"] for r in val],
                color="#d97706", linewidth=1.8, marker=".", label="Validation loss")
        best = min(val, key=lambda r: r["val_loss"])
        ax.scatter([best["step"]], [best["val_loss"]], color="#b45309", zorder=3)
        ax.annotate(f"Best val: {best['val_loss']:.4f} @ {best['step']:,}",
                    (best["step"], best["val_loss"]), xytext=(8, 12),
                    textcoords="offset points", fontsize=10)
    ax.set(title=directory.name, xlabel="Optimizer step", ylabel="Action loss")
    ax.grid(alpha=0.2)
    ax.legend()
    output = directory / "loss_curve.png"
    fig.savefig(output, dpi=160)
    return output


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("This baseline supports cpu and cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run `myvla doctor`; use CPU only for smoke tests.")
    return device


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device.type, dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    )


class StepSampler(Sampler):
    """Uniform frames with replacement; deterministic across checkpoint restarts/prefetch."""

    def __init__(self, n, batch_size, accumulation, start, stop, seed):
        self.n, self.batch_size, self.accumulation = n, batch_size, accumulation
        self.start, self.stop, self.seed = start, stop, seed

    def __len__(self):
        return (self.stop - self.start) * self.accumulation * self.batch_size

    def __iter__(self):
        for micro in range(self.start * self.accumulation, self.stop * self.accumulation):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, micro]))
            yield from rng.integers(self.n, size=self.batch_size).tolist()


def move_batch(batch, device):
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


@torch.no_grad()
def validate(policy, loader, device, precision, max_batches):
    policy.eval()
    total, count = 0.0, 0
    try:
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = move_batch(batch, device)
            with amp_context(device, precision):
                predictions = policy(batch["images"], batch["state"], batch["language"])
                loss, _ = action_loss(predictions, batch["actions"], batch["mask"])
            valid = int(batch["mask"].sum())
            total += loss.item() * valid
            count += valid
    finally:
        policy.train()
    return total / count if count else float("nan")


def rng_state():
    state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path, policy, optimizer, scheduler, scaler, config, stats, episodes, step, best
):
    payload = {
        "format_version": 1,
        "config": config.to_dict(),
        "step": step,
        "best_val": best,
        "stats": {k: v.tolist() for k, v in stats.items()},
        "manifest": manifest(episodes),
        "head": policy.head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
    }
    # Real CLIP is referenced by pinned revision; random smoke weights must be retained.
    if config.model.backend == "tiny":
        payload["tiny_backbone"] = policy.backbone.state_dict()
    temp = Path(str(path) + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return payload


def policy_from_checkpoint(path, device):
    payload = load_checkpoint(path)
    config = from_dict(payload["config"])
    policy = Policy(config.model)
    policy.head.load_state_dict(payload["head"], strict=True)
    if config.model.backend == "tiny":
        policy.backbone.load_state_dict(payload["tiny_backbone"], strict=True)
    policy.to(device).eval()
    stats = {k: np.asarray(v, np.float32) for k, v in payload["stats"].items()}
    return policy, config, stats, payload


def train(config, resume=None, init_checkpoint=None):
    config.validate()
    if resume and init_checkpoint:
        raise ValueError("Use either resume or init_checkpoint, not both")
    t = config.train
    device = resolve_device(t.device)
    if device.type == "cpu" and t.precision != "fp32":
        raise ValueError("CPU runs require precision=fp32")
    if device.type == "cuda" and t.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; choose fp16")
    out = Path(t.output_dir)
    if not resume and any(
        (out / name).exists() for name in ("latest.pt", "best.pt", "metrics.jsonl")
    ):
        raise FileExistsError(f"Existing run: {out}; use --resume or a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(t.seed)
    episodes = scan_episodes(config.data, t.seed)
    saved = load_checkpoint(resume) if resume else None
    initial = load_checkpoint(init_checkpoint) if init_checkpoint else None
    adding_fusion = False
    if initial:
        old = from_dict(initial["config"])
        config.model.revision = old.model.revision
        adding_fusion = old.model.fusion_layers == 0 and config.model.fusion_layers > 0
        if adding_fusion:
            old.model.fusion_layers = config.model.fusion_layers
        if (
            old.model != config.model
            or old.data != config.data
            or initial["manifest"] != manifest(episodes)
        ):
            raise ValueError("Initialization requires the same model and dataset split")
    if saved:
        old = from_dict(saved["config"])
        # Output/device/worker count can change; architecture, schedule and data cannot.
        for key in ("output_dir", "device", "workers", "log_every"):
            setattr(old.train, key, getattr(t, key))
        # Accept a requested branch only if the rest matches; load the saved immutable commit.
        config.model.revision = old.model.revision
        if old.to_dict() != config.to_dict() or saved["manifest"] != manifest(episodes):
            raise ValueError("Resume config/data differs from checkpoint; start a new run")
    stats = (
        {k: np.asarray(v, np.float32) for k, v in saved["stats"].items()}
        if saved
        else compute_stats(episodes, config.data.dataset)
    )
    datasets = {
        s: RobotDataset(episodes, s, config.data, config.model, stats) for s in ("train", "val")
    }
    policy = Policy(config.model).to(device)
    if initial:
        for key, value in stats.items():
            if not np.array_equal(value, np.asarray(initial["stats"][key], np.float32)):
                raise ValueError("Initialization normalization differs from checkpoint")
        incompatible = policy.head.load_state_dict(initial["head"], strict=not adding_fusion)
        if adding_fusion and (
            incompatible.unexpected_keys
            or any(not key.startswith("fusion.") for key in incompatible.missing_keys)
        ):
            raise ValueError("Unexpected keys when initializing fusion")
        if config.model.backend == "tiny":
            policy.backbone.load_state_dict(initial["tiny_backbone"], strict=True)
        (out / "initialization.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(init_checkpoint).resolve()),
                    "checkpoint_step": initial["step"],
                    "optimizer_reset": True,
                    "added_fusion_layers": config.model.fusion_layers if adding_fusion else 0,
                },
                indent=2,
            )
        )
    optimizer = torch.optim.AdamW(policy.head.parameters(), lr=t.lr, weight_decay=t.weight_decay)

    def lr_scale(step):
        if step < t.warmup_steps:
            return (step + 1) / max(t.warmup_steps, 1)
        progress = min(1, (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and t.precision == "fp16")
    start, best = 0, float("inf")
    if saved:
        policy.head.load_state_dict(saved["head"])
        if config.model.backend == "tiny":
            policy.backbone.load_state_dict(saved["tiny_backbone"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start, best = saved["step"], saved["best_val"]
    if start >= t.steps:
        raise ValueError("Checkpoint already reached configured training steps")
    sampler = StepSampler(
        len(datasets["train"]), t.batch_size, t.grad_accumulation, start, t.steps, t.seed
    )
    common = {
        "batch_size": t.batch_size,
        "num_workers": t.workers,
        "pin_memory": device.type == "cuda",
    }
    loader = DataLoader(
        datasets["train"],
        sampler=sampler,
        generator=torch.Generator().manual_seed(t.seed),
        **common,
    )
    # Fixed random ordering prevents a bounded validation pass from covering only task 0.
    val_indices = torch.randperm(
        len(datasets["val"]), generator=torch.Generator().manual_seed(t.seed + 2)
    ).tolist()
    val_loader = DataLoader(
        datasets["val"],
        sampler=val_indices,
        generator=torch.Generator().manual_seed(t.seed + 1),
        **common,
    )
    batches = iter(loader)
    if saved:
        restore_rng(saved["rng"])
    (out / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    (out / "manifest.json").write_text(json.dumps(manifest(episodes), indent=2))
    (out / "stats.json").write_text(json.dumps({k: v.tolist() for k, v in stats.items()}, indent=2))
    print(
        json.dumps(
            {
                "train_frames": len(datasets["train"]),
                "val_frames": len(datasets["val"]),
                "device": str(device),
                "trainable_parameters": sum(p.numel() for p in policy.head.parameters()),
                "backend": config.model.backend,
            }
        ),
        flush=True,
    )
    policy.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with (out / "metrics.jsonl").open("a", buffering=1) as log:
        for step in range(start + 1, t.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            for _ in range(t.grad_accumulation):
                batch = move_batch(next(batches), device)
                batch["images"] = augment_images(batch["images"], t.image_shift, t.color_jitter)
                with amp_context(device, t.precision):
                    predictions = policy(batch["images"], batch["state"], batch["language"])
                    loss, _ = action_loss(predictions, batch["actions"], batch["mask"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                scaler.scale(loss / t.grad_accumulation).backward()
                train_loss += loss.item() / t.grad_accumulation
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(policy.head.parameters(), t.grad_clip)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            if not skipped:
                scheduler.step()
            record = {
                "step": step,
                "loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(norm),
                "optimizer_skipped": skipped,
            }
            improved = False
            if step % t.validate_every == 0 or step == t.steps:
                record["val_loss"] = validate(
                    policy, val_loader, device, t.precision, t.validation_batches
                )
                improved = record["val_loss"] < best
                if improved:
                    best = record["val_loss"]
            if step % t.log_every == 0 or step == start + 1 or "val_loss" in record:
                record["elapsed_seconds"] = time.monotonic() - started
                if device.type == "cuda":
                    record["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                line = json.dumps(record)
                log.write(line + "\n")
                print(line, flush=True)
            if improved:
                save_checkpoint(
                    out / "best.pt",
                    policy,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    stats,
                    episodes,
                    step,
                    best,
                )
            if step % t.save_every == 0 or step == t.steps:
                save_checkpoint(
                    out / "latest.pt",
                    policy,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    stats,
                    episodes,
                    step,
                    best,
                )
            if "val_loss" in record:
                log.flush()
                plot_loss(out)
    return out / "latest.pt"


def add_train_arguments(parser):
    parser.add_argument("--config")
    parser.add_argument(
        "--dataset",
        choices=["libero", "metaworld"],
        help="Select dataset IO defaults; defaults to the YAML/checkpoint setting",
    )
    parser.add_argument("--data-dir", help="Override the training dataset directory")
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint", help="Initialize weights for a new fine-tuning run")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")


def run_training(args, parser, trainer=None):
    if args.config:
        config = load_config(args.config)
    elif args.resume:
        config = from_dict(load_checkpoint(args.resume)["config"])
        # Keep relocated runs in their current directory rather than the saved old path.
        config.train.output_dir = str(Path(args.resume).parent)
    else:
        parser.error("train requires --config or --resume")
    if args.dataset:
        if args.resume and args.dataset != config.data.dataset:
            parser.error("Cannot switch dataset when resuming a checkpoint")
        config = config.for_dataset(args.dataset)
    if args.data_dir is not None:
        config.data.directory = args.data_dir
    for name in ("device", "steps", "workers", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            setattr(config.train, name, value)
    trainer = train if trainer is None else trainer
    return trainer(config, resume=args.resume, init_checkpoint=args.init_checkpoint)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train V1 or V2 on LIBERO or MetaWorld MT50.",
        epilog="Relative config, data, model and output paths use the project root.",
    )
    add_train_arguments(parser)
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    return run_training(args, parser)


if __name__ == "__main__":
    main()
