import pytest
import torch
from torch import nn

from myvla.config import Config
from myvla.ema import HeadEMA
from myvla.train import checkpoint_head


def test_ema_average_has_independent_storage_and_restores_head_after_failure():
    head = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(2)
    ema = HeadEMA(head, 0.75)
    with torch.no_grad():
        head.weight.fill_(6)
    ema.update(head)
    torch.testing.assert_close(ema.module.weight, torch.full_like(head.weight, 3))
    torch.testing.assert_close(head.weight, torch.full_like(head.weight, 6))
    assert not ema.module.weight.requires_grad
    assert ema.updates == 1
    policy = nn.Module()
    policy.head = head
    with pytest.raises(RuntimeError), ema.apply(policy):
        assert policy.head is ema.module
        raise RuntimeError("failed validation")
    assert policy.head is head


def test_legacy_weights_fallback_and_explicit_ema_errors():
    payload = {"head": {"weight": torch.ones(1)}}
    assert checkpoint_head(payload)[1] == "raw"
    with pytest.raises(ValueError, match="no EMA"):
        checkpoint_head(payload, "ema")
    payload["ema_head"] = {"weight": torch.zeros(1)}
    assert checkpoint_head(payload)[1] == "ema"
    assert checkpoint_head(payload, "raw")[0] is payload["head"]
    with pytest.raises(ValueError, match="weights must"):
        checkpoint_head(payload, "invalid")


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan")])
def test_invalid_ema_decay(decay):
    config = Config()
    config.train.ema_decay = decay
    with pytest.raises(ValueError, match="ema_decay"):
        config.validate()
