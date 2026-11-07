import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from myvla.cli import make_smoke_data
from myvla.config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from myvla.data import LiberoDataset, compute_stats, manifest, raw_state, scan_episodes
from myvla.evaluate import Controller, run_episode
from myvla.model import Policy, action_loss
from myvla.preprocess import (
    CLIP_MEAN,
    CLIP_STD,
    augment_images,
    decode_actions,
    normalize_actions,
    prepare_image,
    quat_to_axis_angle,
    state_from_observation,
)
from myvla.train import load_checkpoint, policy_from_checkpoint, train

torch.set_num_threads(1)


@pytest.mark.parametrize(
    ("suite", "patterns"),
    [
        ("libero_spatial", ["libero_spatial/*.hdf5"]),
        ("libero_100", ["libero_90/*.hdf5", "libero_10/*.hdf5"]),
    ],
)
def test_download_retains_existing_data(monkeypatch, tmp_path, suite, patterns):
    from myvla.cli import main

    existing = tmp_path / suite / "existing.hdf5"
    existing.parent.mkdir()
    existing.write_bytes(b"existing data")
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: calls.append(kw))
    monkeypatch.setattr(
        "sys.argv",
        [
            "myvla", "download", "--suite", suite, "--directory", str(tmp_path),
            "--endpoint", "https://example.com", "--max-workers", "1",
        ],
    )
    main()
    assert existing.read_bytes() == b"existing data"
    assert calls == [{
        "repo_id": "yifengzhu-hf/LIBERO-datasets", "repo_type": "dataset",
        "local_dir": str(tmp_path.resolve()), "allow_patterns": patterns,
        "force_download": False, "max_workers": 1, "endpoint": "https://example.com",
    }]


@pytest.fixture
def config(tmp_path):
    make_smoke_data(tmp_path / "data")
    return Config(
        ModelConfig(backend="tiny", image_size=32, hidden_dim=32, layers=1, heads=4),
        DataConfig(directory=str(tmp_path / "data"), val_fraction=0.25),
        TrainConfig(
            output_dir=str(tmp_path / "run"),
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


def test_split_mask_and_training_only_stats(config):
    eps = scan_episodes(config.data, config.train.seed)
    assert manifest(eps) == manifest(scan_episodes(config.data, config.train.seed))
    train_ids = {(e.path, e.demo) for e in eps if e.split == "train"}
    val_ids = {(e.path, e.demo) for e in eps if e.split == "val"}
    assert train_ids and val_ids and train_ids.isdisjoint(val_ids)
    stats = compute_stats(eps)
    states = []
    for ep in eps:
        if ep.split == "train":
            with h5py.File(ep.path) as f:
                states.append(raw_state(f[f"data/{ep.demo}/obs"], slice(None)))
    np.testing.assert_allclose(stats["state_mean"], np.concatenate(states).mean(0), atol=1e-6)
    # Validation corruption must not contaminate normalization statistics.
    for ep in eps:
        if ep.split == "val":
            with h5py.File(ep.path, "r+") as f:
                f[f"data/{ep.demo}/obs/ee_states"][:] = 1e6
    for k, value in compute_stats(eps).items():
        np.testing.assert_array_equal(value, stats[k])
    dataset = LiberoDataset(eps, "train", config.data, config.model, stats)
    last = dataset[len(dataset) - 1]
    assert last["mask"].tolist() == [True] + [False] * 7
    assert last["images"].shape == (2, 3, 32, 32)
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_masked_loss_ignores_padding():
    pred = torch.zeros(2, 8, 7, requires_grad=True)
    target = torch.zeros_like(pred)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, 0] = True
    first, _ = action_loss(pred, target, mask)
    changed = target.clone()
    changed[:, 1:, :6] = 100
    changed[:, 1:, 6] = 1
    second, _ = action_loss(pred, changed, mask)
    torch.testing.assert_close(first, second)
    second.backward()
    assert torch.count_nonzero(pred.grad[:, 1:]) == 0


def test_frozen_backbone_and_output(config):
    policy = Policy(config.model).train()
    assert not policy.backbone.training and not policy.backbone.clip.training
    before = policy.head.output.weight.detach().clone()
    optimizer = torch.optim.AdamW(policy.head.parameters(), lr=0.01)
    pred = policy(torch.randn(2, 2, 3, 32, 32), torch.randn(2, 8), ["bowl", "plate"])
    assert pred.shape == (2, 8, 7)
    loss, _ = action_loss(pred, torch.zeros_like(pred), torch.ones(2, 8, dtype=torch.bool))
    loss.backward()
    optimizer.step()
    assert all(p.grad is None for p in policy.backbone.parameters())
    assert not torch.equal(before, policy.head.output.weight)


def test_v3_updates_vision_while_text_and_unused_clip_projections_stay_frozen(config):
    configs = Path(__file__).resolve().parents[1] / "configs"
    v3 = load_config(configs / "train_cfg_v3.yaml")
    v1 = load_config(configs / "train_cfg_v1.yaml")
    v1.model.train_vision_encoder = True
    assert v3.model == v1.model
    config.model.train_vision_encoder = v3.model.train_vision_encoder
    config.model.fusion_layers = v3.model.fusion_layers
    policy = Policy(config.model).train()
    assert len(policy.head.fusion) == 0
    assert not policy.backbone.use_tokens
    vision = policy.backbone.clip.vision_model
    text = policy.backbone.clip.text_model
    assert vision.training and not text.training
    assert all(p.requires_grad for p in vision.parameters())
    assert all(not p.requires_grad for p in text.parameters())
    before = {name: p.detach().clone() for name, p in vision.named_parameters()}
    text_before = {name: p.detach().clone() for name, p in text.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
    # Repeated language exercises the frozen text cache across backward passes.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        predictions = policy(torch.randn(2, 2, 3, 32, 32), torch.zeros(2, 8), ["bowl"] * 2)
        loss, _ = action_loss(
            predictions, torch.zeros_like(predictions), torch.ones(2, 8, dtype=torch.bool)
        )
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vision.parameters())
        optimizer.step()
    assert any(not torch.equal(before[name], p) for name, p in vision.named_parameters())
    for name, p in text.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, text_before[name], rtol=0, atol=0)
    assert not policy.backbone.clip.visual_projection.weight.requires_grad
    assert not policy.backbone.clip.text_projection.weight.requires_grad
    policy.eval()
    assert not vision.training and not text.training


def test_action_roundtrip_and_online_state():
    stats = {"action_mean": np.arange(6) * 0.01, "action_std": np.full(6, 0.3)}
    actions = np.array(
        [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1], [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1]], np.float32
    )
    encoded = normalize_actions(actions, stats)
    encoded[:, 6] = np.where(encoded[:, 6] > 0.5, 2, -2)
    np.testing.assert_allclose(decode_actions(encoded, stats), actions, atol=1e-6)
    obs = {
        "robot0_eef_pos": [1, 2, 3],
        "robot0_eef_quat": [0, 0, 0, 1],
        "robot0_gripper_qpos": [0.02, -0.02],
    }
    np.testing.assert_allclose(state_from_observation(obs), [1, 2, 3, 0, 0, 0, 0.02, -0.02])
    np.testing.assert_allclose(quat_to_axis_angle([0, 0, 1, 0]), [0, 0, np.pi], atol=1e-6)
    image = np.zeros((8, 8, 3), np.uint8)
    image[0] = 255
    expected = prepare_image(image[::-1].copy(), 8, False)
    torch.testing.assert_close(prepare_image(image, 8, True), expected)


@pytest.mark.parametrize("augmentation", [False, True])
@pytest.mark.parametrize("fusion_layers", [0, 2])
@pytest.mark.parametrize("ema_decay", [0.0, 0.999])
@pytest.mark.parametrize("train_vision", [False, True])
def test_resume_matches_uninterrupted_training(
    config, monkeypatch, tmp_path, augmentation, fusion_layers, ema_decay, train_vision
):
    import myvla.train as training

    config.model.fusion_layers = fusion_layers
    config.train.ema_decay = ema_decay
    config.model.train_vision_encoder = train_vision
    config.train.vision_lr = 1e-5 if train_vision else None
    if augmentation:
        config.train.image_shift = 2
        config.train.color_jitter = 0.1
    reference_path = train(config)
    reference = load_checkpoint(reference_path)
    config.train.output_dir = str(tmp_path / "interrupted")
    original_save = training.save_checkpoint

    def interrupt_after_step_one(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if path.name == "latest.pt":
            raise InterruptedError("Simulated stop after a complete optimizer step")

    with monkeypatch.context() as m:
        m.setattr(training, "save_checkpoint", interrupt_after_step_one)
        with pytest.raises(InterruptedError):
            train(config)
    checkpoint = tmp_path / "interrupted/latest.pt"
    assert load_checkpoint(checkpoint)["step"] == 1
    train(config, resume=checkpoint)
    resumed = load_checkpoint(checkpoint)
    assert resumed["step"] == 3
    for key, value in reference["head"].items():
        torch.testing.assert_close(resumed["head"][key], value, rtol=0, atol=0)
    if ema_decay:
        assert reference["ema_updates"] == resumed["ema_updates"] == 3
        for key, value in reference["ema_head"].items():
            torch.testing.assert_close(resumed["ema_head"][key], value, rtol=0, atol=0)
        assert reference["validation_weights"] == "ema"
    if train_vision:
        for name in (["vision", "ema_vision"] if ema_decay else ["vision"]):
            for key, value in reference[name].items():
                torch.testing.assert_close(resumed[name][key], value, rtol=0, atol=0)
    assert reference["best_val"] == resumed["best_val"]
    policy, _, _, _ = policy_from_checkpoint(checkpoint, torch.device("cpu"))
    assert not policy.training
    chosen = resumed["ema_head"] if ema_decay else resumed["head"]
    for key, value in policy.head.state_dict().items():
        torch.testing.assert_close(value, chosen[key], rtol=0, atol=0)
    if train_vision:
        chosen_vision = resumed["ema_vision"] if ema_decay else resumed["vision"]
        for key, value in policy.backbone.clip.vision_model.state_dict().items():
            torch.testing.assert_close(value, chosen_vision[key], rtol=0, atol=0)
    raw, _, _, _ = policy_from_checkpoint(checkpoint, torch.device("cpu"), weights="raw")
    for key, value in raw.head.state_dict().items():
        torch.testing.assert_close(value, resumed["head"][key], rtol=0, atol=0)
    if train_vision:
        for key, value in raw.backbone.clip.vision_model.state_dict().items():
            torch.testing.assert_close(value, resumed["vision"][key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        train(config)
    records = [
        json.loads(line) for line in (checkpoint.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [1, 2, 3]
    if ema_decay:
        validations = [r for r in records if "val_loss" in r]
        assert validations
        for record in validations:
            assert record["val_loss"] == record["ema_val_loss"]
            assert "raw_val_loss" in record


@pytest.mark.parametrize(
    ("fusion_layers", "train_vision", "legacy"), [(0, False, True), (2, False, False),
                                               (2, True, False)],
)
def test_extend_completed_run_preserves_optimizer_ema_and_original_lr_schedule(
    config, monkeypatch, tmp_path, fusion_layers, train_vision, legacy
):
    import myvla.train as training

    # Keep this integration test focused on training state, not plotting.
    monkeypatch.setattr(training, "plot_loss", lambda _: None)
    config.model.fusion_layers = fusion_layers
    config.model.train_vision_encoder = train_vision
    config.train.ema_decay = 0.9
    config.train.vision_lr = 1e-5 if train_vision else None
    initial_path = train(config)
    original = load_checkpoint(initial_path)
    assert original["step"] == 3
    if legacy:
        original.pop("lr_schedule_steps")
        torch.save(original, initial_path)

    config.train.steps = 7
    config.train.output_dir = str(tmp_path / "extended_reference")
    reference = load_checkpoint(train(config, resume=initial_path))
    assert reference["lr_schedule_steps"] == 3
    assert reference["config"]["train"]["steps"] == 7
    assert reference["ema_updates"] == 7
    assert reference["optimizer"]["param_groups"][0]["lr"] == pytest.approx(config.train.lr * 0.1)
    if train_vision:
        assert reference["optimizer"]["param_groups"][1]["lr"] == pytest.approx(1e-6)
    for state in reference["optimizer"]["state"].values():
        assert state["step"].item() == 7

    config.train.output_dir = str(tmp_path / "extended_interrupted")
    original_save = training.save_checkpoint

    def stop_after_first_extended_step(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if path.name == "latest.pt":
            raise InterruptedError("Stop at step 4")

    with monkeypatch.context() as context:
        context.setattr(training, "save_checkpoint", stop_after_first_extended_step)
        with pytest.raises(InterruptedError):
            train(config, resume=initial_path)
    checkpoint = tmp_path / "extended_interrupted/latest.pt"
    assert load_checkpoint(checkpoint)["step"] == 4
    train(config, resume=checkpoint)
    resumed = load_checkpoint(checkpoint)

    def assert_identical(actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_identical(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                assert_identical(a, b)
        else:
            assert actual == expected

    for key in ("head", "ema_head", "optimizer", "scheduler", "rng", "best_val"):
        assert_identical(resumed[key], reference[key])
    if train_vision:
        for key in ("vision", "ema_vision"):
            assert_identical(resumed[key], reference[key])
    records = [json.loads(line) for line in (checkpoint.parent / "metrics.jsonl")
               .read_text().splitlines()]
    assert [r["step"] for r in records] == [4, 5, 6, 7]
    # A second extension must still retain the original three-step LR horizon.
    config.train.steps = 9
    train(config, resume=checkpoint)
    repeated = load_checkpoint(checkpoint)
    assert repeated["lr_schedule_steps"] == 3
    assert repeated["step"] == repeated["ema_updates"] == 9
    # Reducing a previously saved target remains disallowed.
    config.train.steps = 7
    with pytest.raises(ValueError, match="cannot reduce"):
        train(config, resume=checkpoint)
    # A completed run rejects unrelated hyperparameter changes, even when extended.
    config.train.steps = 10
    config.train.lr *= 2
    with pytest.raises(ValueError, match="Resume config/data differs"):
        train(config, resume=checkpoint)


def test_augmentation_preserves_shape_range_and_inputs():
    pixels = torch.rand(2, 2, 3, 16, 16)
    images = (pixels - CLIP_MEAN) / CLIP_STD
    original = images.clone()
    assert augment_images(images) is images
    torch.manual_seed(17)
    output = augment_images(images, shift=2, color_jitter=0.1)
    torch.manual_seed(17)
    torch.testing.assert_close(output, augment_images(images, 2, 0.1), rtol=0, atol=0)
    torch.testing.assert_close(images, original, rtol=0, atol=0)
    restored = output * CLIP_STD + CLIP_MEAN
    assert restored.shape == pixels.shape
    assert restored.min() >= -1e-6 and restored.max() <= 1 + 1e-6
    assert not torch.equal(output, images)


def test_finetune_initialization_starts_new_schedule(config, tmp_path):
    initial = train(config)
    config.train.output_dir = str(tmp_path / "finetune")
    config.train.steps = 2
    config.train.image_shift = 2
    output = train(config, init_checkpoint=initial)
    payload = load_checkpoint(output)
    assert payload["step"] == 2
    assert payload["scheduler"]["last_epoch"] == 2
    provenance = json.loads((output.parent / "initialization.json").read_text())
    assert provenance["checkpoint_step"] == 3 and provenance["optimizer_reset"]
    assert load_checkpoint(initial)["step"] == 3
    config.train.output_dir = str(tmp_path / "incompatible")
    config.model.chunk_size = 4
    with pytest.raises(ValueError, match="same model and dataset split"):
        train(config, init_checkpoint=initial)


def test_controller_replans_and_clears_episode_queue(config):
    class FakePolicy:
        calls = 0

        def __call__(self, images, state, languages):
            self.calls += 1
            assert images.shape == (1, 2, 3, 32, 32)
            assert languages == ["bowl"]
            return torch.zeros(1, 8, 7)

    obs = {
        "agentview_image": np.zeros((32, 32, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((32, 32, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": [0, 0, 0, 1],
        "robot0_gripper_qpos": np.zeros(2),
    }
    stats = {
        "state_mean": np.zeros(8),
        "state_std": np.ones(8),
        "action_mean": np.zeros(6),
        "action_std": np.ones(6),
    }
    policy = FakePolicy()
    controller = Controller(policy, config, stats, torch.device("cpu"), 2)
    for _ in range(3):
        controller.act(obs, "bowl")
    assert policy.calls == 2
    controller.reset()
    controller.act(obs, "bowl")
    assert policy.calls == 3

    class FakeEnv:
        def reset(self):
            self.steps = 0

        def set_init_state(self, state):
            return obs

        def step(self, action):
            self.steps += 1
            return obs, 0, False, {}

        def check_success(self):
            return self.steps >= 12

    assert run_episode(FakeEnv(), controller, "bowl", None, 10) == (True, 2)


def test_fusion_preserves_initial_policy_and_masks_padding(config):
    from myvla.model import VisionLanguageFusion

    baseline = Policy(config.model).eval()
    config.model.fusion_layers = 2
    fused = Policy(config.model).eval()
    fused.backbone.load_state_dict(baseline.backbone.state_dict())
    fused.head.load_state_dict(baseline.head.state_dict(), strict=False)
    images, state = torch.randn(2, 2, 3, 32, 32), torch.randn(2, 8)
    languages = ["red bowl", "black bowl"]
    torch.testing.assert_close(baseline(images, state, languages), fused(images, state, languages))
    layer = VisionLanguageFusion(32, 4, 0)
    layer.gates.data.fill_(0.2)
    visual, text = torch.randn(2, 5, 32), torch.randn(2, 4, 32)
    padding = torch.tensor([[False, False, True, True], [False, False, False, True]])
    v, t = layer(visual, text, padding)
    poisoned = text.clone()
    poisoned[padding] = 1000
    v2, t2 = layer(visual, poisoned, padding)
    torch.testing.assert_close(v, v2)
    torch.testing.assert_close(t[~padding], t2[~padding])
    (v.square().mean() + t[~padding].square().mean()).backward()
    for attention in (layer.visual_attention, layer.text_attention):
        assert attention.in_proj_weight.grad.abs().sum() > 0
    fused(images, state, languages).square().mean().backward()
    assert all(p.grad is None for p in fused.backbone.parameters())
    assert fused.head.fusion[0].gates.grad.abs().sum() > 0


def test_fusion_initialization_and_checkpoint_reload(config, tmp_path):
    initial = train(config)
    config.model.fusion_layers = 2
    config.train.output_dir = str(tmp_path / "fusion")
    checkpoint = train(config, init_checkpoint=initial)
    policy, saved_config, _, _ = policy_from_checkpoint(checkpoint, torch.device("cpu"))
    assert saved_config.model.fusion_layers == 2
    assert policy.head.fusion[0].gates.abs().sum() > 0
