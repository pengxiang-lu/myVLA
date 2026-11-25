from dataclasses import replace
from pathlib import Path

import pytest
import torch

from myvla.cli import make_smoke_data
from myvla.config import Config, DataConfig, ModelConfig, TrainConfig, from_dict, load_config
from myvla.model import DiffusionActionHead, Policy
from myvla.preprocess import decode_actions
from myvla.train import load_checkpoint, policy_from_checkpoint, train

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


def tiny_model(**kwargs):
    return ModelConfig(
        backend="tiny", image_size=16, hidden_dim=16, heads=2, layers=1,
        chunk_size=3, fusion_layers=6, train_vision_encoder=True, dropout=0,
        action_head="diffusion", diffusion_train_steps=20, diffusion_inference_steps=4, **kwargs,
    )


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_all"])
def test_v33_only_changes_action_head(suite):
    base = load_config(ROOT / "configs" / suite / "train_cfg_v2_2.yaml")
    v3 = load_config(ROOT / "configs" / suite / "train_cfg_v3_3.yaml")
    assert replace(v3.model, action_head="regression") == base.model
    assert v3.data == base.data
    assert replace(v3.train, output_dir=base.train.output_dir) == base.train
    assert v3.for_dataset("metaworld").train.output_dir.endswith("metaworld_mt50_v3_3")
    legacy = base.to_dict()
    legacy["model"].pop("action_head")
    legacy["model"].pop("flow_steps")
    legacy["model"].pop("diffusion_train_steps")
    legacy["model"].pop("diffusion_inference_steps")
    assert from_dict(legacy).model.action_head == "regression"


@pytest.mark.parametrize("field,value", [
    ("diffusion_train_steps", 1), ("diffusion_train_steps", True),
    ("diffusion_train_steps", 1.5), ("diffusion_inference_steps", 0),
    ("diffusion_inference_steps", 21), ("diffusion_inference_steps", True),
])
def test_diffusion_steps_validation(field, value):
    with pytest.raises(ValueError, match="diffusion"):
        Config(model=replace(tiny_model(), **{field: value})).validate()


def test_noise_objective_and_forward_process(monkeypatch):
    head = DiffusionActionHead(tiny_model(), 32, 32)
    monkeypatch.setattr(head, "encode_context", lambda *args: torch.zeros(2, 4, 16))
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.full_like(x, 2))
    monkeypatch.setattr(torch, "randint", lambda high, shape, **kw: torch.full(shape, 5, **kw))
    target = torch.zeros(2, 3, 7)
    target[1, :, 6] = 1
    signed = target.clone()
    signed[..., 6] = target[..., 6] * 2 - 1
    def predict(noisy, timesteps, memory):
        alpha = head.alpha_bar[5]
        torch.testing.assert_close(noisy, alpha.sqrt() * signed + (1-alpha).sqrt() * 2)
        return torch.full_like(noisy, 2)
    monkeypatch.setattr(head, "predict_noise", predict)
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


@pytest.mark.parametrize("steps", [1, 4, 20])
def test_ddim_recovers_clean_actions_with_oracle(monkeypatch, steps):
    head = DiffusionActionHead(replace(tiny_model(), diffusion_inference_steps=steps), 32, 32)
    clean = torch.randn(2, 3, 7) * 2  # Must not clip normalized actions to [-1, 1].
    calls = []
    def predict(actions, timesteps, memory):
        calls.append(timesteps[0].item())
        alpha = head.alpha_bar[timesteps, None, None]
        return (actions - alpha.sqrt() * clean) / (1-alpha).sqrt()
    monkeypatch.setattr(head, "predict_noise", predict)
    result = head.sample(torch.zeros(2, 4, 16), torch.randn_like(clean))
    torch.testing.assert_close(result, clean, atol=1e-4, rtol=1e-4)
    assert len(calls) == steps and calls[0] == 19
    assert all(a > b for a, b in zip(calls, calls[1:]))
    assert torch.all(head.alpha_bar > 0) and torch.all(head.alpha_bar < 1)
    assert torch.all(head.alpha_bar[1:] < head.alpha_bar[:-1])


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
        assert isinstance(policy.head, DiffusionActionHead)
        assert restored.model.diffusion_train_steps == 20
        assert restored.model.diffusion_inference_steps == 4
        assert payload["inference_weights"] == weights
        with torch.no_grad():
            result = policy(torch.zeros(1, 2, 3, 16, 16), torch.zeros(1, 8), ["pick"])
        assert result.shape == (1, 3, 7) and torch.isfinite(result).all()
