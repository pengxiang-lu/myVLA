"""Profile full ViT-B/16 architecture + action head with random weights, no downloads.

This checks CUDA/AMP/memory, not pretrained features, learning quality or LIBERO success.
Run: uv run --locked python scripts/profile_cuda.py
"""

import json
import time

import torch
from transformers import CLIPConfig, CLIPModel

from myvla.config import ModelConfig
from myvla.model import ActionHead, action_loss


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    config = ModelConfig()
    clip_config = CLIPConfig(vision_config={"patch_size": 16, "image_size": 224})
    backbone = CLIPModel(clip_config).requires_grad_(False).eval().to(device)
    head = (
        ActionHead(
            config, clip_config.vision_config.hidden_size, clip_config.text_config.hidden_size
        )
        .to(device)
        .train()
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    images = torch.randn(2, 2, 3, 224, 224, device=device)
    state = torch.randn(2, 8, device=device)
    tokens = torch.tensor([[49406, 100, 200, 49407]] * 2, device=device)
    targets = torch.zeros(2, 8, 7, device=device)
    mask = torch.ones(2, 8, dtype=torch.bool, device=device)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                vision = backbone.vision_model(images.flatten(0, 1)).last_hidden_state[:, 1:]
                vision = vision.reshape(2, 2, -1, clip_config.vision_config.hidden_size).float()
                text = backbone.text_model(input_ids=tokens).pooler_output.float()
            predictions = head(vision, text, state)
            loss, _ = action_loss(predictions, targets, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite CUDA probe loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    assert all(p.grad is None for p in backbone.parameters())
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "micro_batch": 2,
                "image_size": 224,
                "cameras": 2,
                "chunk_size": 8,
                "decoder_layers": 4,
                "hidden_dim": 256,
                "trainable_parameters": sum(p.numel() for p in head.parameters()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "seconds_for_3_steps": time.monotonic() - started,
                "finite_loss": loss.item(),
                "note": "Random full-size CLIP weights; memory/AMP check only, not a benchmark.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
