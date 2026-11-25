from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import SiglipConfig, SiglipModel

from myvla.cli import make_smoke_data
from myvla.config import Config, DataConfig, TrainConfig, load_config
from myvla.model import ActionHead, FlowMatchingActionHead, Policy
from myvla.preprocess import CLIP_MEAN, CLIP_STD
from myvla.train import load_checkpoint, policy_from_checkpoint, train

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


@pytest.fixture
def offline_siglip(monkeypatch):
    # Real SigLIP layers, miniature dimensions; no network/pretrained benchmark claims.
    template = SiglipModel(SiglipConfig(
        text_config={"vocab_size": 128, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2,
                     "max_position_embeddings": 64},
        vision_config={"hidden_size": 16, "intermediate_size": 32,
                       "num_hidden_layers": 1, "num_attention_heads": 2,
                       "image_size": 224, "patch_size": 16},
    ))

    class Tokenizer:
        def __call__(self, language, **kwargs):
            assert kwargs["padding"] == "max_length" and kwargs["max_length"] == 64
            ids = [3 + ord(c) % 125 for c in language[:63]] + [1]
            ids += [1] * (64 - len(ids))
            # Return mask deliberately: adapter must discard it to match pretraining.
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.tensor([[1] * len(language[:63]) +
                                                     [0] * (64 - len(language[:63]))])}

    monkeypatch.setattr("myvla.model.SiglipModel.from_pretrained",
                        lambda *a, **kw: deepcopy(template))
    monkeypatch.setattr("myvla.model.AutoTokenizer.from_pretrained",
                        lambda *a, **kw: Tokenizer())
    config = load_config(ROOT / "configs/libero_spatial/train_cfg_v3_1.yaml")
    config.model.hidden_dim = 16
    config.model.heads = 2
    config.model.layers = 1
    config.model.dropout = 0
    config.model.chunk_size = 3
    return config


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_all"])
def test_v3_1_config_is_v22_with_siglip(suite):
    base = load_config(ROOT / "configs" / suite / "train_cfg_v2_2.yaml")
    v3_1 = load_config(ROOT / "configs" / suite / "train_cfg_v3_1.yaml")
    assert replace(v3_1.model, backbone=base.model.backbone, revision=base.model.revision,
                   backend=base.model.backend) == base.model
    assert v3_1.data == base.data
    assert replace(v3_1.train, output_dir=base.train.output_dir) == base.train
    assert v3_1.model.action_head == "regression"
    assert v3_1.for_dataset("metaworld").train.output_dir.endswith("metaworld_mt50_v3_1")


@pytest.mark.parametrize("action_head", ["regression", "flow_matching"])
@pytest.mark.parametrize("dataset", ["libero", "metaworld"])
def test_siglip_normalization_tokens_and_gradients(offline_siglip, action_head, dataset):
    config = offline_siglip.for_dataset(dataset)
    config.model.action_head = action_head
    m = config.model
    policy = Policy(m).train()
    expected_head = FlowMatchingActionHead if action_head == "flow_matching" else ActionHead
    assert type(policy.head) is expected_head
    captured = []
    hook = policy.backbone.clip.vision_model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs["pixel_values"].detach()),
        with_kwargs=True,
    )
    rgb = torch.rand(2, m.cameras, 3, 224, 224)
    images = (rgb - CLIP_MEAN) / CLIP_STD
    vision, pooled, tokens, padding = policy.backbone(images, ["pick", "move bowl"])
    hook.remove()
    torch.testing.assert_close(captured[0], (rgb.flatten(0, 1) - .5) / .5)
    assert vision.shape == (2, m.cameras, 196, 16)  # All patches preserved: no CLS to remove.
    assert pooled.shape == (2, 16) and tokens.shape == (2, 64, 16)
    assert not padding.any()
    assert set(policy.backbone.tokenize("pick", "cpu")) == {"input_ids"}
    assert torch.equal(pooled, policy.backbone.encode_text(["pick", "move bowl"], "cpu")[0])
    loss, _ = policy.compute_loss(
        images, torch.zeros(2, m.state_dim), ["pick", "move bowl"],
        torch.zeros(2, m.chunk_size, m.action_dim), torch.ones(2, m.chunk_size, dtype=torch.bool),
    )
    loss.backward()
    assert policy.backbone.clip.vision_model.embeddings.patch_embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.backbone.clip.text_model.parameters())
    assert policy.head.output.weight.grad.abs().sum() > 0
    assert not policy.backbone.clip.text_model.training
    assert policy.backbone.clip.vision_model.training


@pytest.mark.parametrize("action_head", ["regression", "flow_matching"])
def test_siglip_checkpoint_resume_ema_and_uncached_inference(
    offline_siglip, tmp_path, monkeypatch, action_head,
):
    import importlib.util
    spec = importlib.util.spec_from_file_location("benchmark_siglip",
                                                ROOT / "scripts/measure_inference_time.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setattr("myvla.train.plot_loss", lambda _: None)
    make_smoke_data(tmp_path / "data")
    offline_siglip.model.action_head = action_head
    offline_siglip.model.flow_steps = 2
    config = Config(
        model=offline_siglip.model, data=DataConfig(directory=str(tmp_path / "data")),
        train=TrainConfig(output_dir=str(tmp_path / "run"), device="cpu", precision="fp32",
                          batch_size=1, grad_accumulation=1, workers=0, steps=2,
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
        assert restored.model.backend == "siglip"
        assert restored.model.action_head == action_head
        expected_head = FlowMatchingActionHead if action_head == "flow_matching" else ActionHead
        assert type(policy.head) is expected_head
        assert payload["inference_weights"] == weights
        images, state = torch.zeros(1, 2, 3, 224, 224), torch.zeros(1, 8)
        with torch.no_grad():
            torch.manual_seed(7)
            cached = policy(images, state, ["pick"])
            text_inputs = benchmark.prepare_text(policy.backbone, "pick", "cpu")
            torch.manual_seed(7)
            uncached = benchmark.predict_chunk(policy, images, state, text_inputs)
        torch.testing.assert_close(cached, uncached)
        assert cached.shape == (1, 3, 7) and torch.isfinite(cached).all()


@pytest.mark.parametrize("suite", ["libero_spatial", "libero_all"])
def test_v3_2_combines_siglip_and_flow_matching(suite):
    configs = ROOT / "configs" / suite
    v3 = load_config(configs / "train_cfg_v3.yaml")
    v3_1 = load_config(configs / "train_cfg_v3_1.yaml")
    v3_2 = load_config(configs / "train_cfg_v3_2.yaml")
    assert replace(v3_2.model, action_head=v3_1.model.action_head) == v3_1.model
    assert replace(v3_2.model, backend=v3.model.backend, backbone=v3.model.backbone,
                   revision=v3.model.revision) == v3.model
    assert v3_2.data == v3_1.data == v3.data
    assert v3_2.train.output_dir == f"outputs/train/{suite}_v3_2"
    assert replace(v3_2.train, output_dir=v3_1.train.output_dir) == v3_1.train
    assert v3_2.for_dataset("metaworld").train.output_dir == "outputs/train/metaworld_mt50_v3_2"
    assert v3_2.model.flow_steps == 10


def test_download_siglip_uses_pinned_weights(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_siglip", ROOT / "scripts/download_siglip.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module, "snapshot_download", lambda **kw: calls.append(kw))
    module.main(["--directory", str(tmp_path), "--endpoint", "https://example.com"])
    assert calls[0]["revision"] == load_config(
        ROOT / "configs/libero_all/train_cfg_v3_1.yaml").model.revision
    assert "model.safetensors" in calls[0]["allow_patterns"]
    assert "pytorch_model.bin" not in calls[0]["allow_patterns"]


def test_real_siglip_tokenizer_round_trip(tmp_path):
    import sentencepiece as spm
    from transformers import AutoTokenizer, SiglipTokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("pick up the bowl\nmove the cup to the left\nopen the drawer\n" * 10)
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(tmp_path / "spiece"), vocab_size=32,
        hard_vocab_limit=False, bos_id=-1, eos_id=1, unk_id=2, pad_id=-1,
        minloglevel=2,
    )
    tokenizer = SiglipTokenizer(
        vocab_file=str(tmp_path / "spiece.model"), model_input_names=["input_ids"]
    )
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    restored = AutoTokenizer.from_pretrained(tmp_path / "tokenizer", local_files_only=True)
    inputs = restored("PICK up the bowl", padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt")
    assert inputs["input_ids"].shape == (1, 64)
    assert inputs["input_ids"][0, -1].item() == restored.eos_token_id
    assert "attention_mask" not in inputs
    lower = restored("pick up the bowl", padding="max_length", max_length=64,
                     return_tensors="pt")
    torch.testing.assert_close(inputs["input_ids"], lower["input_ids"])
