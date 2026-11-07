import importlib
from pathlib import Path

import pytest

from myvla import evaluate as libero_evaluation
from myvla import metaworld_evaluate

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    return importlib.import_module(f"myvla.{name}")


@pytest.mark.parametrize("override", [None, "outputs/train/continued"])
def test_resume_uses_relocated_checkpoint_directory(monkeypatch, tmp_path, override):
    script = load_script("train")
    config = script.load_config(ROOT / "configs/train_cfg_v1.yaml").to_dict()
    config["train"]["output_dir"] = "runs/old_location"
    config["train"]["save_every"] = 1000
    monkeypatch.setattr(script, "load_checkpoint", lambda _: {"config": config})
    captured = []
    monkeypatch.setattr(script, "train", lambda config, **kw: captured.append(config))
    monkeypatch.chdir(tmp_path)
    args = ["--resume", "outputs/train/libero_spatial_v1/latest.pt"]
    if override is not None:
        args.extend(["--output-dir", override])
    script.main(args)
    assert captured[0].train.output_dir == (override or "outputs/train/libero_spatial_v1")
    assert captured[0].train.save_every == 2000


def test_train_script_from_another_directory(monkeypatch, tmp_path):
    script = load_script("train")
    captured = []
    monkeypatch.setattr(script, "train", lambda config, **kw: captured.append(config.validate()))
    monkeypatch.chdir(tmp_path)
    script.main(
        [
            "--config",
            "configs/train_cfg_v2.yaml",
            "--dataset",
            "metaworld",
            "--steps",
            "7",
            "--data-dir",
            "data/custom_mt50",
            "--output-dir",
            "outputs/train/custom_mt50",
        ]
    )
    config = captured[0]
    assert Path.cwd() == ROOT
    assert config.model.fusion_layers == 6
    assert config.data.dataset == "metaworld"
    assert config.data.directory == "data/custom_mt50"
    assert config.train.steps == 7
    assert config.train.output_dir == "outputs/train/custom_mt50"


@pytest.mark.parametrize(
    "dataset,episodes,execute,horizon",
    [
        ("libero", 10, 12, None),
        ("metaworld", 50, 1, 500),
    ],
)
def test_evaluation_script_selects_checkpoint_dataset(
    monkeypatch,
    tmp_path,
    dataset,
    episodes,
    execute,
    horizon,
):
    script = load_script("evaluate")
    monkeypatch.setattr(
        script, "load_checkpoint", lambda _: {"config": {"data": {"dataset": dataset}}}
    )
    captured = []
    target, name = (
        (metaworld_evaluate, "evaluate_metaworld")
        if dataset == "metaworld"
        else (libero_evaluation, "evaluate")
    )
    monkeypatch.setattr(target, name, lambda **kw: captured.append(kw))
    monkeypatch.chdir(tmp_path)
    script.main(
        ["--checkpoint", "outputs/train/check/best.pt", "--output-dir", "outputs/eval/check"]
    )
    args = captured[0]
    assert Path.cwd() == ROOT
    assert args["episodes"] == episodes
    assert args["execute_steps"] == execute
    assert args["horizon"] == horizon
    if dataset == "metaworld":
        assert args["protocol"] == "lerobot"
    else:
        assert args["suite"] == "libero_spatial"


def test_evaluation_script_preserves_explicit_options(monkeypatch, tmp_path):
    script = load_script("evaluate")
    monkeypatch.setattr(
        script, "load_checkpoint", lambda _: {"config": {"data": {"dataset": "metaworld"}}}
    )
    captured = []
    monkeypatch.setattr(metaworld_evaluate, "evaluate_metaworld", lambda **kw: captured.append(kw))
    monkeypatch.chdir(tmp_path)
    script.main(
        [
            "--checkpoint",
            "best.pt",
            "--output-dir",
            "eval",
            "--dataset",
            "metaworld",
            "--episodes",
            "2",
            "--execute-steps",
            "3",
            "--horizon",
            "80",
            "--tasks",
            "0",
            "1",
            "--protocol",
            "benchmark",
            "--videos",
            "--device",
            "cpu",
            "--seed",
            "7",
        ]
    )
    args = captured[0]
    assert args["episodes"] == 2 and args["execute_steps"] == 3 and args["horizon"] == 80
    assert args["tasks"] == [0, 1] and args["seed"] == 7
    assert args["protocol"] == "benchmark" and args["videos"] and args["device_name"] == "cpu"


@pytest.mark.parametrize(
    "options",
    [
        ["--dataset", "libero"],
        ["--suite", "libero_spatial"],
        ["--libero-root", "/tmp/libero"],
    ],
)
def test_evaluation_script_rejects_wrong_dataset_options(monkeypatch, tmp_path, options):
    script = load_script("evaluate")
    monkeypatch.setattr(
        script, "load_checkpoint", lambda _: {"config": {"data": {"dataset": "metaworld"}}}
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        script.main(["--checkpoint", "best.pt", "--output-dir", "eval", *options])
    assert exc.value.code == 2
