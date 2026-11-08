"""Exercise real evaluator control flow without simulator/model dependencies."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from myvla import evaluate as libero_eval
from myvla import metaworld_evaluate as meta_eval
from myvla.config import Config


@pytest.mark.parametrize("backend", ["libero", "metaworld_lerobot", "metaworld_benchmark"])
@pytest.mark.parametrize("videos", [False, True])
def test_evaluator_resumes_committed_episodes_and_matches_uninterrupted(
    tmp_path, monkeypatch, backend, videos,
):
    meta = backend.startswith("metaworld")
    module = meta_eval if meta else libero_eval
    config = Config().for_dataset("metaworld") if meta else Config()
    checkpoint = tmp_path / "checkpoint_2000.pt"
    checkpoint.write_bytes(b"test weights")
    payload = {
        "step": 2000, "inference_weights": "ema", "config": config.to_dict(), "manifest": [],
    }
    monkeypatch.setattr(module, "policy_from_checkpoint", lambda *a, **kw: (
        object(), config, {}, payload,
    ))
    monkeypatch.setattr(module, "version", lambda _: "test")
    created, closed, calls = [], [], []
    video_paths, video_closes = [], []
    imageio = ModuleType("imageio")
    imageio_v2 = ModuleType("imageio.v2")

    def get_writer(path, **kwargs):
        from pathlib import Path

        path = Path(path)
        assert path.parent.is_dir() and path.parent.name == "videos"
        path.touch()
        video_paths.append(path)
        return SimpleNamespace(
            append_data=lambda frame: None, close=lambda: video_closes.append(path),
        )

    imageio_v2.get_writer = get_writer
    imageio.v2 = imageio_v2
    monkeypatch.setitem(sys.modules, "imageio", imageio)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio_v2)

    class Env:
        def __init__(self, name):
            self.name = name
            created.append(name)

        def seed(self, seed):
            pass

        def set_task(self, task):
            self.task = task

        def close(self):
            closed.append(self.name)

    if meta:
        monkeypatch.setattr(meta_eval, "MetaWorldController", lambda *a: object())
        monkeypatch.setattr(meta_eval, "import_metaworld", lambda: SimpleNamespace(
            MT50=lambda seed: object(),
        ))
        monkeypatch.setattr(meta_eval, "make_env", lambda name, *a: (
            Env(name), [SimpleNamespace(data=bytes([i])) for i in range(5)],
        ))

        def run_episode(env, controller, language, seed, *a, **kw):
            calls.append((env.name, seed))
            if len(calls) == 5 and interrupt[0]:
                raise KeyboardInterrupt()
            return bool(seed % 2), 10, 1.0

        kwargs = dict(tasks=[0, 1], protocol=backend.removeprefix("metaworld_"), horizon=20)
        evaluate = meta_eval.evaluate_metaworld
    else:
        monkeypatch.setattr(libero_eval, "Controller", lambda *a: object())
        monkeypatch.setattr(libero_eval, "setup_libero", lambda _: {"source_root": "test"})
        suite = SimpleNamespace(
            n_tasks=2,
            get_task=lambda i: SimpleNamespace(name=f"task_{i}", language=f"task_{i}"),
            get_task_bddl_file_path=lambda i: f"task_{i}",
        )
        parent = ModuleType("libero")
        package = ModuleType("libero.libero")
        benchmark = ModuleType("libero.libero.benchmark")
        envs = ModuleType("libero.libero.envs")
        benchmark.get_benchmark_dict = lambda: {"libero_spatial": lambda **kw: suite}
        envs.OffScreenRenderEnv = lambda bddl_file_name, **kw: Env(bddl_file_name)
        package.benchmark = benchmark
        parent.libero = package
        for name, value in {
            "libero": parent, "libero.libero": package,
            "libero.libero.benchmark": benchmark, "libero.libero.envs": envs,
        }.items():
            monkeypatch.setitem(sys.modules, name, value)
        monkeypatch.setattr(libero_eval, "load_init_states", lambda *a: (np.arange(5), "sha"))

        def run_episode(env, controller, language, state, *a, **kw):
            calls.append((env.name, int(state)))
            if len(calls) == 5 and interrupt[0]:
                raise KeyboardInterrupt()
            return bool(state % 2), 10

        kwargs = dict(suite="libero_spatial", tasks=[0, 1], horizon=20)
        evaluate = libero_eval.evaluate
    monkeypatch.setattr(module, "run_episode", run_episode)
    interrupt = [True]
    common = dict(
        checkpoint=checkpoint, episodes=3, device_name="cpu", execute_steps=1,
        videos=videos, **kwargs,
    )
    output = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        evaluate(output=output, **common)
    out = tmp_path / "resumed/ckpt2000/seed42"
    progress = json.loads((out / "progress.json").read_text())
    assert sorted(map(len, progress.values())) == [1, 3]
    assert not (out / "log.json").exists()
    assert len(closed) == len(created) == 2
    first_completed = calls[:4]
    interrupt[0] = False
    result = evaluate(output=output, **common)
    # One interrupted episode is retried; completed episodes (including failures) are skipped.
    assert len(calls) == 7 and calls[4] == calls[5]
    assert calls[:4] == first_completed
    assert len(created) == len(closed) == 3
    lines = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert len(lines) == 7
    assert lines[0]["type"] == "evaluation_metadata"
    assert len({(r["task"], r["episode"]) for r in lines[1:]}) == 6
    assert lines[0]["evaluation"]["weights"] == "ema"
    assert lines[0]["config"] == payload["config"]
    assert not (out / "config.yaml").exists()
    assert not (out / "evaluation.json").exists()
    assert json.loads((out / "log.json").read_text()) == result
    assert set(result) == {"success_rate", "total_success_rate"}
    # Already completed runs must return their summary without creating any environments.
    assert evaluate(output=output, **common) == result
    assert len(calls) == 7 and len(created) == 3
    assert (out / "videos").exists() == videos
    if videos:
        assert len(list((out / "videos").glob("*.mp4"))) == 6
        assert len(video_paths) == len(video_closes) == 7
    reference = evaluate(output=tmp_path / "reference", **common)
    assert result == reference
    reference_progress = tmp_path / "reference/ckpt2000/seed42/progress.json"
    assert json.loads(reference_progress.read_text()) == json.loads(
        (out / "progress.json").read_text()
    )
    with pytest.raises(ValueError, match="seed directory"):
        evaluate(output=out, **dict(common, seed=7))
    original = (out / "progress.json").read_bytes()
    evaluate(output=output, **dict(common, seed=7))
    assert (output / "ckpt2000/seed7/log.json").exists()
    assert (out / "progress.json").read_bytes() == original
