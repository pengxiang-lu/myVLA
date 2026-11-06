"""EMA of the action head and, optionally, the trainable visual encoder."""

from contextlib import contextmanager
from copy import deepcopy

import torch


class HeadEMA:
    def __init__(self, head, decay, vision=None):
        self.module = deepcopy(head).requires_grad_(False).eval()
        self.vision = deepcopy(vision).requires_grad_(False).eval() if vision is not None else None
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, head, vision=None):
        pairs = [(self.module, head)]
        if self.vision is not None:
            if vision is None:
                raise ValueError("Visual EMA requires the current visual encoder")
            pairs.append((self.vision, vision))
        for target, module in pairs:
            source = module.state_dict()
            for name, averaged in target.state_dict().items():
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
        original_vision = None
        if self.vision is not None:
            original_vision = policy.backbone.clip.vision_model
            policy.backbone.clip.vision_model = self.vision
        try:
            yield
        finally:
            policy.head = original
            self.module.eval()
            if original_vision is not None:
                policy.backbone.clip.vision_model = original_vision
                self.vision.eval()
