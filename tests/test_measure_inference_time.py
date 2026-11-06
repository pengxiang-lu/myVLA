import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from myvla.config import Config, ModelConfig
from myvla.model import Policy
from myvla.preprocess import decode_actions

spec = importlib.util.spec_from_file_location(
    "measure_inference_time",
    Path(__file__).resolve().parents[1] / "scripts/measure_inference_time.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("fusion_layers", [0, 2])
def test_uncached_inference_matches_policy_and_reports_checkpoint_horizon(
    tmp_path, fusion_layers, monkeypatch
):
    torch.set_num_threads(1)
    config = Config(model=ModelConfig(
        backend="tiny", image_size=16, hidden_dim=32, layers=1, heads=4,
        fusion_layers=fusion_layers, chunk_size=12,
    ))
    policy = Policy(config.model).eval()
    language = "pick up the bowl"
    images = torch.zeros(1, 2, 3, 16, 16)
    state = torch.zeros(1, 8)
    text = benchmark.prepare_text(policy.backbone, language, torch.device("cpu"))
    with torch.inference_mode():
        expected = policy(images, state, [language])
        actual = benchmark.predict_chunk(policy, images, state, text)
    torch.testing.assert_close(actual, expected)
    stats = {
        "action_mean": [0.1] * 6, "action_std": [0.5] * 6,
        "state_mean": [0.0] * 8, "state_std": [1.0] * 8,
    }
    checkpoint = tmp_path / "tiny.pt"
    torch.save({
        "format_version": 1, "config": config.to_dict(), "step": 123,
        "head": policy.head.state_dict(), "tiny_backbone": policy.backbone.state_dict(),
        "stats": stats, "manifest": [{"language": language}],
    }, checkpoint)
    monkeypatch.chdir(tmp_path)
    report = tmp_path / "report.json"
    result = benchmark.main([
        "--checkpoint", str(checkpoint), "--device", "cpu",
        "--warmup", "1", "--runs", "2", "--output", str(report),
    ])
    assert result["output_shape"] == [1, 12, 7]
    assert result["checkpoint_step"] == 123
    assert len(result["samples_ms"]) == 2
    assert result["latency_ms"]["median"] > 0
    assert json.loads(report.read_text()) == result


@pytest.mark.parametrize("action_dim", [4, 7])
def test_device_decoding_matches_rollout(action_dim):
    continuous_dim = 4 if action_dim == 4 else 6
    stats = {
        "action_mean": np.linspace(-0.2, 0.2, continuous_dim, dtype=np.float32),
        "action_std": np.linspace(0.1, 0.9, continuous_dim, dtype=np.float32),
    }
    predictions = torch.randn(1, 12, action_dim) * 3
    predictions[0, 0, -1] = 0  # Same gripper threshold at zero.
    actual = benchmark.decode_chunk(
        predictions, torch.from_numpy(stats["action_mean"]), torch.from_numpy(stats["action_std"])
    )
    np.testing.assert_allclose(actual.numpy(), decode_actions(predictions.numpy(), stats))
