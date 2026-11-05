import json

import h5py
import numpy as np
import pytest
import torch

from myvla.cli import make_smoke_data
from myvla.config import Config, DataConfig, ModelConfig, TrainConfig
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
def test_resume_matches_uninterrupted_training(config, monkeypatch, tmp_path, augmentation):
    import myvla.train as training

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
    assert reference["best_val"] == resumed["best_val"]
    policy, _, _, _ = policy_from_checkpoint(checkpoint, torch.device("cpu"))
    assert not policy.training
    with pytest.raises(FileExistsError):
        train(config)
    records = [
        json.loads(line) for line in (checkpoint.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == [1, 2, 3]


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
