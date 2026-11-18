from dataclasses import replace
from pathlib import Path

import pytest
import torch

from myvla.cli import make_smoke_data
from myvla.config import Config, DataConfig, ModelConfig, TrainConfig, from_dict, load_config
from myvla.model import FlowMatchingActionHead, Policy
from myvla.preprocess import decode_actions
from myvla.train import load_checkpoint, policy_from_checkpoint, train

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def tiny_model(**kwargs):
    return ModelConfig(
        backend="tiny", image_size=16, hidden_dim=16, heads=2, layers=1,
        chunk_size=3, fusion_layers=6, train_vision_encoder=True, dropout=0,
        action_head="flow_matching", flow_steps=2, **kwargs,
    )


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_all"])
def test_v3_only_changes_action_head(suite):
    base = load_config(ROOT / "configs" / suite / "train_cfg_v2_2.yaml")
    v3 = load_config(ROOT / "configs" / suite / "train_cfg_v3.yaml")
    assert replace(v3.model, action_head="regression") == base.model
    assert v3.data == base.data
    assert replace(v3.train, output_dir=base.train.output_dir) == base.train
    assert v3.for_dataset("metaworld").train.output_dir.endswith("metaworld_mt50_v3")
    legacy = base.to_dict()
    legacy["model"].pop("action_head")
    legacy["model"].pop("flow_steps")
    assert from_dict(legacy).model.action_head == "regression"


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_flow_steps_validation(value):
    with pytest.raises(ValueError, match="flow_steps"):
        Config(model=replace(tiny_model(), flow_steps=value)).validate()


def test_linear_path_and_gripper_target(monkeypatch):
    head = FlowMatchingActionHead(tiny_model(), 32, 32)
    memory = torch.zeros(2, 4, 16)
    monkeypatch.setattr(head, "encode_context", lambda *a: memory)
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.full_like(x, 2))
    monkeypatch.setattr(torch, "rand", lambda n, **kw: torch.full((n,), .25, **kw))
    target = torch.zeros(2, 3, 7)
    target[1, :, 6] = 1
    signed = target.clone()
    signed[..., 6] = target[..., 6] * 2 - 1

    def velocity(noisy, time, context):
        torch.testing.assert_close(noisy, .75 * 2 + .25 * signed)
        torch.testing.assert_close(time, torch.full((2,), .25))
        return signed - 2

    monkeypatch.setattr(head, "velocity", velocity)
    loss, _ = head.loss(None, None, None, target, torch.ones(2, 3, dtype=torch.bool))
    assert loss == 0


@pytest.mark.parametrize("meta", [False, True])
def test_mask_gradients_sampling_and_decode(meta):
    config = tiny_model(**({"action_dim": 4, "state_dim": 4, "cameras": 1} if meta else {}))
    policy = Policy(config)
    images = torch.randn(2, config.cameras, 3, 16, 16)
    state = torch.randn(2, config.state_dim)
    targets = torch.randn(2, 3, config.action_dim)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    changed = targets.clone()
    changed[~mask] = 10000
    torch.manual_seed(7)
    loss, _ = policy.compute_loss(images, state, ["pick", "move"], targets, mask)
    torch.manual_seed(7)
    other, _ = policy.compute_loss(images, state, ["pick", "move"], changed, mask)
    torch.testing.assert_close(loss, other, rtol=0, atol=0)
    loss.backward()
    assert policy.head.action_proj.weight.grad.abs().sum() > 0
    assert policy.head.time_proj[0].weight.grad.abs().sum() > 0
    assert policy.backbone.clip.vision_model.embeddings.patch_embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.backbone.clip.text_model.parameters())
    policy.eval()
    with torch.no_grad():
        torch.manual_seed(9)
        result = policy(images, state, ["pick", "move"])
        torch.manual_seed(9)
        repeat = policy(images, state, ["pick", "move"])
    assert result.shape == targets.shape and torch.isfinite(result).all()
    torch.testing.assert_close(result, repeat, rtol=0, atol=0)
    import numpy as np
    dim = 4 if meta else 6
    decoded = decode_actions(result.numpy(), {"action_mean": np.zeros(dim),
                                             "action_std": np.ones(dim)})
    assert np.isfinite(decoded).all()
    if not meta:
        assert set(np.unique(decoded[..., 6])) <= {-1, 1}


def test_euler_integrates_velocity(monkeypatch):
    head = FlowMatchingActionHead(tiny_model(), 32, 32)
    calls = []

    def velocity(actions, time, memory):
        calls.append(time[0].item())
        return torch.ones_like(actions) * 3

    monkeypatch.setattr(head, "velocity", velocity)
    noise = torch.randn(2, 3, 7)
    result = head.sample(torch.zeros(2, 4, 16), noise)
    torch.testing.assert_close(result, noise + 3)
    assert calls == [0, .5]


def test_training_resume_ema_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("myvla.train.plot_loss", lambda _: None)
    make_smoke_data(tmp_path / "data")
    config = Config(
        model=tiny_model(), data=DataConfig(directory=str(tmp_path / "data")),
        train=TrainConfig(output_dir=str(tmp_path / "run"), device="cpu", precision="fp32",
                          batch_size=2, grad_accumulation=1, workers=0, steps=2,
                          warmup_steps=1, save_every=1, log_every=1, ema_decay=.9),
    )
    final = train(config)
    reference = load_checkpoint(final)
    config.train.output_dir = str(tmp_path / "resumed")
    final = train(config, resume=final.parent / "checkpoint_1.pt")
    resumed = load_checkpoint(final)
    for section in ["head", "vision", "ema_head", "ema_vision"]:
        for name, value in reference[section].items():
            torch.testing.assert_close(value, resumed[section][name], rtol=0, atol=0)
    for weights in ["raw", "ema"]:
        policy, restored, _, payload = policy_from_checkpoint(final, torch.device("cpu"), weights)
        assert isinstance(policy.head, FlowMatchingActionHead)
        assert restored.model.flow_steps == 2
        assert payload["inference_weights"] == weights
        with torch.no_grad():
            result = policy(torch.zeros(1, 2, 3, 16, 16), torch.zeros(1, 8), ["pick"])
        assert result.shape == (1, 3, 7) and torch.isfinite(result).all()
