from pathlib import Path

import pytest

from myvla import cli
from myvla import train as training
from myvla.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_dataset_selection_preserves_architecture_and_training(version):
    original = load_config(CONFIGS / f"train_cfg_{version}.yaml")
    meta = original.for_dataset("metaworld")
    assert original.data.dataset == "libero"
    assert meta.model.fusion_layers == original.model.fusion_layers
    assert (meta.model.state_dim, meta.model.action_dim, meta.model.cameras) == (4, 4, 1)
    assert meta.data.directory == "data/metaworld_mt50"
    assert not meta.data.flip_vertical
    assert meta.train.output_dir == f"outputs/train/metaworld_mt50_{version}"
    original_train, meta_train = vars(original.train).copy(), vars(meta.train).copy()
    original_train.pop("output_dir")
    meta_train.pop("output_dir")
    assert original_train == meta_train
    libero = meta.for_dataset("libero")
    assert libero.to_dict() == original.to_dict()


def test_training_cli_dataset_and_directory_overrides(monkeypatch):
    captured = []
    monkeypatch.setattr(cli, "train", lambda config, **kw: captured.append(config.validate()))
    monkeypatch.setattr(
        "sys.argv",
        [
            "myvla",
            "train",
            "--config",
            str(CONFIGS / "train_cfg_v1.yaml"),
            "--dataset",
            "metaworld",
            "--data-dir",
            "/tmp/custom-mt50",
            "--output-dir",
            "outputs/train/custom-mt50",
            "--steps",
            "10",
        ],
    )
    cli.main()
    config = captured[0]
    assert config.data.dataset == "metaworld"
    assert config.model.fusion_layers == 0
    assert config.data.directory == "/tmp/custom-mt50"
    assert config.train.output_dir == "outputs/train/custom-mt50"
    assert config.train.steps == 10


def test_resume_cannot_switch_dataset(monkeypatch):
    config = load_config(CONFIGS / "train_cfg_v2.yaml")
    monkeypatch.setattr(training, "load_checkpoint", lambda _: {"config": config.to_dict()})
    monkeypatch.setattr(
        "sys.argv",
        [
            "myvla",
            "train",
            "--resume",
            "existing.pt",
            "--dataset",
            "metaworld",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
