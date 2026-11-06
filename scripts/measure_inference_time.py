"""Measure resident-input, complete-action-chunk latency without a simulator."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myvla.train import amp_context, policy_from_checkpoint, resolve_device, seed_everything

ROOT = Path(__file__).resolve().parents[1]


def prepare_text(backbone, language, device):
    if backbone.tokenizer is not None:
        return backbone.tokenizer(
            language, return_tensors="pt", truncation=True, max_length=77
        ).to(device)
    digest = hashlib.sha256(language.encode()).digest()[:16]
    ids = [1] + [3 + b % 125 for b in digest] + [2]
    return {"input_ids": torch.tensor([ids], device=device)}


def predict_chunk(policy, images, state, text_inputs):
    """Same features as Policy.forward, but run text encoding every time on device.

    All input tensors are already resident. This deliberately bypasses the CPU
    text cache used by normal rollouts, so the timing includes language encoding.
    """
    backbone = policy.backbone
    vision = backbone.clip.vision_model(
        pixel_values=images.flatten(0, 1)
    ).last_hidden_state[:, 1:]
    vision = vision.reshape(*images.shape[:2], -1, backbone.vision_dim).float()
    text = backbone.clip.text_model(**text_inputs)
    if not backbone.use_tokens:
        return policy.head(vision, text.pooler_output.float(), state)
    tokens = text.last_hidden_state.float()
    # Batch size one: tokenizer produces no padding for this single instruction.
    padding = torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    return policy.head(vision, text.pooler_output.float(), state, tokens, padding)


def decode_chunk(predictions, mean, std):
    """Device-side equivalent of preprocess.decode_actions (no CPU transfer)."""
    predictions = predictions.float()
    if mean.numel() == 4:
        return (predictions * std + mean).clamp(-1, 1)
    motion = predictions[..., :6] * std + mean
    gripper = torch.where(predictions[..., 6:] >= 0, 1.0, -1.0)
    return torch.cat((motion, gripper), dim=-1).clamp(-1, 1)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(args):
    device = resolve_device(args.device)
    precision = args.precision
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU measurements require --precision fp32")
    if device.type == "cuda" and precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support BF16")
    seed_everything(42)
    policy, config, stats, payload = policy_from_checkpoint(
        args.checkpoint, device, weights=args.weights
    )
    step = payload["step"]
    selected_weights = payload["inference_weights"]
    language = args.language or payload["manifest"][0]["language"]
    del payload  # Optimizer/checkpoint data is not needed for inference.
    m = config.model
    # Fixed synthetic, already-normalized observations; no data IO or simulator.
    images = torch.zeros(1, m.cameras, 3, m.image_size, m.image_size, device=device)
    state = torch.zeros(1, m.state_dim, device=device)
    text_inputs = prepare_text(policy.backbone, language, device)
    mean = torch.as_tensor(stats["action_mean"], device=device)
    std = torch.as_tensor(stats["action_std"], device=device)

    def infer():
        with amp_context(device, precision):
            prediction = predict_chunk(policy, images, state, text_inputs)
        return decode_chunk(prediction, mean, std)

    for _ in range(args.warmup):
        actions = infer()
        synchronize(device)
        del actions
    durations = []
    for _ in range(args.runs):
        synchronize(device)
        started = time.perf_counter()
        actions = infer()
        synchronize(device)
        durations.append((time.perf_counter() - started) * 1000)
        # Validation deliberately outside the timed region.
        if not torch.isfinite(actions).all().item():
            raise FloatingPointError("Non-finite output actions")
        shape = list(actions.shape)
        del actions
    median = float(np.median(durations))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": step,
        "weights": selected_weights,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "precision": precision,
        "parameter_dtype": str(next(policy.parameters()).dtype),
        "backend": m.backend,
        "batch_size": 1,
        "image_size": m.image_size,
        "cameras": m.cameras,
        "fusion_layers": m.fusion_layers,
        "train_vision_encoder": m.train_vision_encoder,
        "chunk_size": m.chunk_size,
        "output_shape": shape,
        "language": language,
        "text_tokens": text_inputs["input_ids"].shape[1],
        "text_cache": False,
        "synthetic_observations": True,
        "warmup": args.warmup,
        "runs": args.runs,
        "latency_ms": {
            "median": median,
            "mean": float(np.mean(durations)),
            "p95": float(np.percentile(durations, 95)),
            "min": min(durations),
            "max": max(durations),
        },
        "chunk_predictions_per_second": 1000 / median,
        "samples_ms": durations,
        "scope": "resident inputs -> vision + text encoding + head + decoded chunk on device",
        "excluded": "input preparation/transfer, CPU output copy, loading, environment execution",
        "note": "Steady-state eager inference; full text encoder each run; no CUDA Graph/compile.",
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
        help="Compute precision; parameters remain FP32 (default: fp32, like evaluation)",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--language", help="Default: first checkpoint manifest instruction")
    parser.add_argument("--output", help="Optional JSON report; overwritten if it exists")
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.runs < 1:
        parser.error("--warmup and --runs must be positive")
    os.chdir(ROOT)
    return benchmark(args)


if __name__ == "__main__":
    main()
