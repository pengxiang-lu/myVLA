import io
import json

import h5py
import numpy as np
import pytest
import torch
from PIL import Image

from myvla.config import Config, DataConfig, ModelConfig, TrainConfig
from myvla.data import LiberoDataset, compute_stats, scan_episodes
from myvla.metaworld_data import convert_metaworld, task_definitions, task_language
from myvla.metaworld_evaluate import MetaWorldController, render_rgb, run_episode
from myvla.model import action_loss
from myvla.preprocess import decode_actions, normalize_actions
from myvla.train import policy_from_checkpoint, train


@pytest.fixture
def raw_mt50(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "raw"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "total_episodes": 6,
        "total_frames": 24,
        "features": {
            "observation.state": {"shape": [4]},
            "action": {"shape": [4]},
            "observation.image": {"dtype": "image"},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = []
    for episode in range(6):
        for t in range(4):
            image = np.full((32, 32, 3), episode * 20 + t, np.uint8)
            blob = io.BytesIO()
            Image.fromarray(image).save(blob, format="PNG")
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": t,
                    "task_id": [episode // 3],
                    "observation.state": [float(episode), 0.2, 0.3, 0.4],
                    "action": [2.0 if t == 3 else 0.1 * t, -0.2, 0.3, 0.25],
                    "observation.image": {"bytes": blob.getvalue(), "path": None},
                }
            )
    # Cross a file boundary within a trajectory.
    for i, chunk in enumerate([rows[:10], rows[10:]]):
        pq.write_table(pa.Table.from_pylist(chunk), root / f"data/chunk-000/file-{i:03d}.parquet")
    pq.write_table(
        pa.Table.from_pylist([{"episode_index": i, "length": 4} for i in range(6)]),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    return root


@pytest.fixture
def mt50_config(raw_mt50, tmp_path):
    output = tmp_path / "converted"
    report = convert_metaworld(raw_mt50, output, task_ids=[0, 1], image_size=32)
    assert report["frames"] == 24 and not report["full_mt50"]
    assert report["clipped_action_frames"] == 6
    return Config(
        ModelConfig(
            backend="tiny",
            image_size=32,
            hidden_dim=32,
            heads=4,
            layers=1,
            fusion_layers=1,
            state_dim=4,
            action_dim=4,
            cameras=1,
        ),
        DataConfig(directory=str(output), dataset="metaworld", flip_vertical=False),
        TrainConfig(
            output_dir=str(tmp_path / "run"),
            device="cpu",
            precision="fp32",
            batch_size=2,
            grad_accumulation=2,
            steps=2,
            workers=0,
            warmup_steps=1,
            save_every=1,
            log_every=1,
        ),
    ).validate()


def test_mt50_import_all_training_stats_and_windows(mt50_config):
    config = mt50_config
    episodes = scan_episodes(config.data, 42)
    assert len(episodes) == 6
    assert all(e.split == "train" for e in episodes)
    stats = compute_stats(episodes, "metaworld")
    state_values = []
    for ep in episodes:
        with h5py.File(ep.path, "r+") as f:
            state_values.extend(f[f"data/{ep.demo}/obs/state"][:])
    np.testing.assert_allclose(stats["state_mean"], np.mean(state_values, axis=0, dtype=np.float64))
    for key, value in compute_stats(episodes, "metaworld").items():
        np.testing.assert_array_equal(value, stats[key])
    dataset = LiberoDataset(episodes, "train", config.data, config.model, stats)
    sample = dataset[len(dataset) - 1]
    assert sample["images"].shape == (1, 3, 32, 32)
    assert sample["state"].shape == (4,)
    assert sample["actions"].shape == (8, 4)
    assert sample["mask"].tolist() == [True] + [False] * 7
    action = np.array([[0.2, -0.3, 0.1, 0.25]], np.float32)
    np.testing.assert_allclose(decode_actions(normalize_actions(action, stats), stats), action)


def test_mt50_partial_import_is_atomic(raw_mt50, tmp_path):
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="Incomplete MT50"):
        convert_metaworld(raw_mt50, output, image_size=32)
    assert not output.exists()
    meta = raw_mt50 / "meta/episodes/chunk-000/file-000.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.Table.from_pylist([{"episode_index": i, "length": 5} for i in range(6)]), meta
    )
    with pytest.raises(ValueError, match="Incomplete trajectory"):
        convert_metaworld(raw_mt50, output, task_ids=[0, 1], image_size=32)
    assert not output.exists()


def test_mt50_continuous_loss_and_padding():
    target = torch.randn(2, 3, 4)
    pred = target.clone().requires_grad_()
    mask = torch.tensor([[True, False, False], [True, True, False]])
    loss, parts = action_loss(pred, target, mask)
    assert loss == 0 and set(parts) == {"continuous"}
    changed = target.clone()
    changed[~mask] = 1e9
    torch.testing.assert_close(action_loss(pred, changed, mask)[0], loss)
    altered = target.clone()
    altered[..., 3] += 0.2
    loss = action_loss(pred, altered, mask)[0]
    loss.backward()
    assert pred.grad[mask][:, 3].abs().sum() > 0


def test_mt50_training_resume_and_inference(mt50_config, tmp_path, monkeypatch):
    import myvla.train as training

    config = mt50_config
    reference_path = train(config)
    reference, _, _, payload = policy_from_checkpoint(reference_path, torch.device("cpu"))
    assert payload["step"] == 2
    config.train.output_dir = str(tmp_path / "resume")
    save = training.save_checkpoint

    def stop(path, *args, **kwargs):
        save(path, *args, **kwargs)
        if path.name.startswith("checkpoint_"):
            raise InterruptedError()

    with monkeypatch.context() as m:
        m.setattr(training, "save_checkpoint", stop)
        with pytest.raises(InterruptedError):
            train(config)
    checkpoint = tmp_path / "resume/checkpoint_1.pt"
    resumed_path = train(config, resume=checkpoint)
    resumed, _, _, _ = policy_from_checkpoint(resumed_path, torch.device("cpu"))
    for a, b in zip(reference.head.parameters(), resumed.head.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert reference(torch.randn(2, 1, 3, 32, 32), torch.randn(2, 4), ["a", "b"]).shape == (2, 8, 4)


def test_mt50_controller_reset_state_and_success():
    class FakePolicy:
        calls = 0

        def __call__(self, images, state, language):
            self.calls += 1
            assert images.shape == (1, 1, 3, 224, 224)
            torch.testing.assert_close(state, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
            return torch.zeros(1, 8, 4)

    class FakeEnv:
        count = 0

        def seed(self, seed):
            self.last_seed = seed

        def render(self):
            return np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        def reset(self, seed):
            self.count = 0
            return np.arange(1, 40, dtype=np.float32), {}

        def step(self, action):
            self.count += 1
            assert action.shape == (4,)
            return np.arange(1, 40, dtype=np.float32), 0, False, False, {"success": self.count == 2}

    policy, env = FakePolicy(), FakeEnv()
    stats = {
        "state_mean": np.zeros(4, np.float32),
        "state_std": np.ones(4, np.float32),
        "action_mean": np.zeros(4, np.float32),
        "action_std": np.ones(4, np.float32),
    }
    config = Config()
    controller = MetaWorldController(policy, config, stats, torch.device("cpu"), 4)
    assert run_episode(env, controller, "reach", 42, 10)[:2] == (True, 2)
    assert policy.calls == 1
    assert run_episode(env, controller, "reach", 43, 10)[:2] == (True, 2)
    assert env.last_seed == 43
    assert policy.calls == 2  # Must clear the unused action queue between episodes.
    np.testing.assert_array_equal(render_rgb(env), np.flip(env.render(), (0, 1)))


def test_mt50_task_identity_and_schema_guards():
    definitions = task_definitions()
    assert set(definitions["task_ids"].values()) == set(range(50))
    assert task_language("push-v3") != task_language("push-back-v3")
    with pytest.raises(ValueError, match="requires state_dim"):
        Config(data=DataConfig(dataset="metaworld", flip_vertical=False)).validate()
