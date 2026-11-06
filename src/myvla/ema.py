"""EMA of the trainable action head; the frozen CLIP is shared."""

from contextlib import contextmanager
from copy import deepcopy

import torch


class HeadEMA:
    def __init__(self, head, decay):
        self.module = deepcopy(head).requires_grad_(False).eval()
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, head):
        source = head.state_dict()
        for name, averaged in self.module.state_dict().items():
            current = source[name].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(current, alpha=1 - self.decay)
            else:
                averaged.copy_(current)
        self.updates += 1

    @contextmanager
    def apply(self, policy):
        original = policy.head
        policy.head = self.module
        try:
            yield
        finally:
            policy.head = original
            self.module.eval()
