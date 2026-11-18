"""Run the four-suite evaluator and resume flow with lightweight simulator stand-ins."""

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from myvla import evaluate as evaluation
from myvla.config import Config

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@pytest.mark.parametrize("horizon", [None, 17])
def test_all_suites_keep_separate_progress_and_resume(tmp_path, monkeypatch, horizon):
    config = Config()
    checkpoint = tmp_path / "checkpoint_2000.pt"
    checkpoint.write_bytes(b"test weights")
    payload = {"step": 2000, "inference_weights": "ema", "config": config.to_dict()}
    monkeypatch.setattr(evaluation, "policy_from_checkpoint", lambda *a, **kw: (
        object(), config, {}, payload,
    ))
    monkeypatch.setattr(evaluation, "Controller", lambda *a: object())
    monkeypatch.setattr(evaluation, "setup_libero", lambda _: {"source_root": "test"})
    monkeypatch.setattr(evaluation, "version", lambda _: "test")
    monkeypatch.setattr(evaluation, "load_init_states", lambda *a: (np.arange(4), "sha"))
    created, closed, calls = [], [], []

    class Env:
        def __init__(self, name):
            self.name = name
            created.append(name)

        def seed(self, seed):
            pass

        def close(self):
            closed.append(self.name)

    def make_suite(name):
        return SimpleNamespace(
            n_tasks=2,
            # Deliberately reuse task names across suites to exercise output isolation.
            get_task=lambda i: SimpleNamespace(name=f"task_{i}", language=f"task_{i}"),
            get_task_bddl_file_path=lambda i: f"{name}/task_{i}",
        )

    parent = ModuleType("libero")
    package = ModuleType("libero.libero")
    benchmark = ModuleType("libero.libero.benchmark")
    envs = ModuleType("libero.libero.envs")
    benchmark.get_benchmark_dict = lambda: {
        name: (lambda name=name, **kw: make_suite(name)) for name in SUITES
    }
    envs.OffScreenRenderEnv = lambda bddl_file_name, **kw: Env(bddl_file_name)
    package.benchmark = benchmark
    parent.libero = package
    for name, value in {
        "libero": parent, "libero.libero": package,
        "libero.libero.benchmark": benchmark, "libero.libero.envs": envs,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    interrupt = [True]

    def run_episode(env, controller, language, state, horizon, **kwargs):
        calls.append((env.name, int(state), horizon))
        if len(calls) == 7 and interrupt[0]:
            raise KeyboardInterrupt()
        return bool(state % 2), 10

    monkeypatch.setattr(evaluation, "run_episode", run_episode)
    options = dict(
        checkpoint=checkpoint, suite="libero_all", tasks=None, episodes=2,
        device_name="cpu", execute_steps=1, horizon=horizon,
    )
    output = tmp_path / "eval"
    with pytest.raises(KeyboardInterrupt):
        evaluation.evaluate(output=output, **options)
    assert (output / "ckpt2000/seed42/libero_spatial/log.json").exists()
    assert not (output / "ckpt2000/seed42/libero_object/log.json").exists()
    interrupt[0] = False
    # The explicit checkpoint/seed output path must work for multi-suite resume too.
    result = evaluation.evaluate(output=output / "ckpt2000/seed42", **options)
    assert len(calls) == 17 and calls[6] == calls[7]
    assert created == closed
    assert tuple(result["suite_success_rate"]) == SUITES
    assert set(result["success_rate"]) == {
        f"{suite}/task_{task}" for suite in SUITES for task in range(2)
    }
    assert result["total_success_rate"] == pytest.approx(
        sum(result["success_rate"].values()) / 8
    )
    for suite in SUITES:
        out = output / "ckpt2000/seed42" / suite
        progress = json.loads((out / "progress.json").read_text())
        assert sum(map(len, progress.values())) == 4
        summary = json.loads((out / "log.json").read_text())
        assert summary["total_success_rate"] == result["suite_success_rate"][suite]
        lines = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
        assert len(lines) == 5
        metadata = lines[0]["evaluation"]
        assert metadata["suite"] == suite
        assert metadata["horizon"] == (horizon or evaluation.HORIZONS[suite])
        assert all(call[2] == metadata["horizon"] for call in calls if call[0].startswith(suite))
    before = len(created)
    assert evaluation.evaluate(output=output, **options) == result
    assert len(calls) == 17 and len(created) == before
    assert evaluation.evaluate(output=tmp_path / "reference", **options) == result


def test_all_suites_preserve_shared_options(monkeypatch, tmp_path):
    calls = []

    def evaluate_suite(**kwargs):
        calls.append(kwargs)
        return {"success_rate": {"task": 0.5}, "total_success_rate": 0.5}

    monkeypatch.setattr(evaluation, "_evaluate_suite", evaluate_suite)
    options = dict(
        checkpoint="checkpoint.pt", tasks=[1], episodes=3, device_name="cpu",
        execute_steps=2, output=tmp_path, seed=7, horizon=13, videos=True,
        libero_root="source", weights="raw",
    )
    result = evaluation.evaluate(suite="libero_all", **options)
    assert calls == [dict(options, suite=suite, separate_suite=True) for suite in SUITES]
    assert result["total_success_rate"] == 0.5


def test_direct_evaluation_rejects_libero_90():
    with pytest.raises(ValueError, match="Unsupported LIBERO suite"):
        evaluation.evaluate("missing.pt", "libero_90", None, 1, "cpu", 1, "eval")
