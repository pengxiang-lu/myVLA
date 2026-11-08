import pytest

from myvla.eval_utils import prepare_eval_output


def test_output_uses_checkpoint_step_and_keeps_parent(tmp_path):
    out = prepare_eval_output(tmp_path / "eval" / "v1", 27000)
    assert out == tmp_path / "eval" / "v1" / "ckpt27000" / "seed42"
    assert out.is_dir()
    assert prepare_eval_output(out, 27000) == out


def test_execution_length_or_incorrect_step_in_prefix_does_not_hide_actual_step(tmp_path):
    assert prepare_eval_output(tmp_path / "v1_12", 27000) == tmp_path / "v1_12/ckpt27000/seed42"
    assert prepare_eval_output(tmp_path / "v1_25000", 4000).parent.name == "ckpt4000"


def test_steps_share_experiment_parent_and_wrong_explicit_step_is_rejected(tmp_path):
    first = prepare_eval_output(tmp_path / "v1", 2000)
    second = prepare_eval_output(tmp_path / "v1", 4000)
    assert first.parent.parent == second.parent.parent == tmp_path / "v1"
    assert first != second
    with pytest.raises(ValueError, match="must be ckpt4000"):
        prepare_eval_output(first, 4000)


def test_seed_directories_are_separate_and_explicit_paths_are_idempotent(tmp_path):
    out = prepare_eval_output(tmp_path / "v1", 10000, 42)
    other = prepare_eval_output(tmp_path / "v1", 10000, 7)
    assert out == tmp_path / "v1/ckpt10000/seed42"
    assert other == out.with_name("seed7")
    assert prepare_eval_output(out.parent, 10000, 42) == out
    assert prepare_eval_output(out, 10000, 42) == out
    with pytest.raises(ValueError, match="seed directory"):
        prepare_eval_output(out, 10000, 7)


@pytest.mark.parametrize("filename", ["log.json", "results.json", "episodes.jsonl"])
def test_existing_reports_are_not_overwritten(tmp_path, filename):
    out = prepare_eval_output(tmp_path / "v1", 4000)
    report = out / filename
    report.write_text("existing result")
    with pytest.raises(FileExistsError, match="ckpt4000"):
        prepare_eval_output(tmp_path / "v1", 4000)
    assert report.read_text() == "existing result"


def make_progress(tmp_path, **overrides):
    from myvla.eval_utils import EvaluationProgress

    checkpoint = tmp_path / "checkpoint.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"model weights")
    args = dict(
        out=prepare_eval_output(tmp_path / "eval", 2000), checkpoint=checkpoint,
        config={"model": {"hidden_dim": 32}}, metadata={"weights": "ema", "seed": 42},
        task_names=["BinFill", "StopCube"], episodes=2,
    )
    args.update(overrides)
    return EvaluationProgress(**args)


def test_progress_persists_success_and_failure_and_final_schema(tmp_path):
    import json

    progress = make_progress(tmp_path)
    progress.record("BinFill", 0, True, {"steps": 10})
    progress.record("BinFill", 1, False, {"steps": 220})
    assert json.loads((progress.out / "progress.json").read_text()) == {
        "BinFill": {"0": True, "1": False}, "StopCube": {},
    }
    with pytest.raises(ValueError, match="incomplete"):
        progress.finish()
    resumed = make_progress(tmp_path)
    assert resumed.done("BinFill", 1) and resumed.task_done("BinFill")
    resumed.record("StopCube", 0, True, {})
    resumed.record("StopCube", 1, True, {})
    result = resumed.finish()
    assert result == {
        "success_rate": {"BinFill": 0.5, "StopCube": 1.0}, "total_success_rate": 0.75,
    }
    assert json.loads((progress.out / "log.json").read_text()) == result
    header = json.loads((progress.out / "episodes.jsonl").read_text().splitlines()[0])
    assert header["type"] == "evaluation_metadata"
    assert header["evaluation"]["weights"] == "ema"
    assert header["config"] == {"model": {"hidden_dim": 32}}
    assert not (progress.out / "config.yaml").exists()
    assert not (progress.out / "evaluation.json").exists()
    assert make_progress(tmp_path).finish() == result


@pytest.mark.parametrize("change", ["checkpoint", "weights", "seed", "config", "episodes"])
def test_resume_rejects_incompatible_runs(tmp_path, change):
    progress = make_progress(tmp_path)
    progress.record("BinFill", 0, False, {})
    if change == "checkpoint":
        (tmp_path / "checkpoint.pt").write_bytes(b"different model at the same step")
        kwargs = {}
    elif change in {"weights", "seed"}:
        kwargs = {"metadata": {"weights": "raw", "seed": 42}}
        if change == "seed":
            kwargs["metadata"] = {"weights": "ema", "seed": 7}
    elif change == "episodes":
        # Production metadata also records episodes_per_task.
        kwargs = {"metadata": {"weights": "ema", "seed": 42, "episodes_per_task": 3},
                  "episodes": 3}
    else:
        kwargs = {"config": {"model": {"hidden_dim": 64}}}
    with pytest.raises(ValueError, match="differ"):
        make_progress(tmp_path, **kwargs)
    assert progress.done("BinFill", 0)


@pytest.mark.parametrize("value", [{"0": 1}, {"-1": True}, {"2": False}, []])
def test_invalid_progress_is_not_overwritten(tmp_path, value):
    import json

    progress = make_progress(tmp_path)
    path = progress.out / "progress.json"
    path.write_text(json.dumps({"BinFill": value}))
    before = path.read_text()
    with pytest.raises(ValueError):
        make_progress(tmp_path)
    assert path.read_text() == before


def test_resume_reconciles_uncommitted_duplicate_and_torn_episode_logs(tmp_path):
    import json

    progress = make_progress(tmp_path)
    progress.record("BinFill", 0, False, {"steps": 220})
    path = progress.out / "episodes.jsonl"
    committed = path.read_text()
    with path.open("a") as log:
        log.write(committed)
        log.write(json.dumps({"task": "BinFill", "episode": 1, "success": True}) + "\n")
        log.write('{"task":')
    resumed = make_progress(tmp_path)
    assert path.read_text() == committed
    assert resumed.done("BinFill", 0) and not resumed.done("BinFill", 1)


def test_interrupted_atomic_progress_write_keeps_previous_results(tmp_path, monkeypatch):
    from pathlib import Path

    progress = make_progress(tmp_path)
    progress.record("BinFill", 0, True, {})
    before = (progress.out / "progress.json").read_text()
    replace = Path.replace

    def interrupt_replace(path, target):
        if path.name == "progress.json.tmp":
            raise KeyboardInterrupt()
        return replace(path, target)

    with monkeypatch.context() as context:
        context.setattr(Path, "replace", interrupt_replace)
        with pytest.raises(KeyboardInterrupt):
            progress.record("BinFill", 1, False, {})
    assert (progress.out / "progress.json").read_text() == before
    resumed = make_progress(tmp_path)
    assert resumed.done("BinFill", 0) and not resumed.done("BinFill", 1)
    resumed.record("BinFill", 1, False, {})
    assert len((progress.out / "episodes.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("legacy_format", ["yaml", "json"])
def test_legacy_metadata_moves_to_episode_log_and_keeps_progress(tmp_path, legacy_format):
    import json

    import yaml

    progress = make_progress(tmp_path)
    progress.record("BinFill", 0, False, {})
    path = progress.out / "episodes.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    header = json.loads(lines[0])
    path.write_text("".join(lines[1:]))
    config_path = progress.out / "config.yaml"
    combined = dict(header["config"], evaluation=header["evaluation"])
    config_path.write_text(yaml.safe_dump(combined))
    legacy = progress.out / "evaluation.json"
    if legacy_format == "json":
        config_path.write_text(yaml.safe_dump(header["config"]))
        legacy.write_text(json.dumps(dict(header["evaluation"], config=header["config"])))
    before = (progress.out / "progress.json").read_text()
    resumed = make_progress(tmp_path)
    assert resumed.done("BinFill", 0)
    assert not legacy.exists() and not config_path.exists()
    assert (progress.out / "progress.json").read_text() == before
    assert json.loads(path.read_text().splitlines()[0]) == header
    assert make_progress(tmp_path).done("BinFill", 0)
