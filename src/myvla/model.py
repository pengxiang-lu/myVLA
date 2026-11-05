import hashlib

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPConfig, CLIPModel, CLIPTokenizerFast


class FrozenCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backend = config.backend
        if self.backend == "tiny":
            # A real, randomly initialized miniature CLIP for offline CPU tests only.
            clip_config = CLIPConfig(
                text_config={
                    "vocab_size": 128,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "max_position_embeddings": 32,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "pad_token_id": 0,
                },
                vision_config={
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "image_size": config.image_size,
                    "patch_size": 4,
                },
                projection_dim=32,
            )
            self.clip = CLIPModel(clip_config)
            self.tokenizer = None
        else:
            self.clip = CLIPModel.from_pretrained(
                config.backbone, revision=config.revision, attn_implementation="sdpa"
            )
            # Store the resolved commit rather than a moving branch in checkpoints.
            config.revision = self.clip.config._commit_hash or config.revision
            self.tokenizer = CLIPTokenizerFast.from_pretrained(
                config.backbone, revision=config.revision
            )
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.vision_dim = self.clip.config.vision_config.hidden_size
        self.text_dim = self.clip.config.text_config.hidden_size
        self.text_cache = {}

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images, languages):
        batch, cameras = images.shape[:2]
        vision = self.clip.vision_model(pixel_values=images.flatten(0, 1)).last_hidden_state
        # Drop CLS; each patch already carries CLIP spatial position information.
        vision = vision[:, 1:].reshape(batch, cameras, -1, self.vision_dim)
        device = images.device
        texts = []
        for language in languages:
            if language not in self.text_cache:
                if self.tokenizer is None:
                    digest = hashlib.sha256(language.encode()).digest()[:16]
                    ids = [1] + [3 + b % 125 for b in digest] + [2]
                    inputs = {"input_ids": torch.tensor([ids], device=device)}
                else:
                    inputs = self.tokenizer(
                        language, return_tensors="pt", truncation=True, max_length=77
                    ).to(device)
                text = self.clip.text_model(**inputs).pooler_output[0]
                self.text_cache[language] = text.float().cpu()
            texts.append(self.text_cache[language])
        return vision.float(), torch.stack(texts).to(device)


class ActionHead(nn.Module):
    def __init__(self, config, vision_dim, text_dim):
        super().__init__()
        d = config.hidden_dim
        self.vision_proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, d))
        self.text_proj = nn.Linear(text_dim, d)
        self.state_proj = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.camera_embedding = nn.Parameter(torch.randn(1, 2, 1, d) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(1, 2, d) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(1, config.chunk_size, d) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d,
            config.heads,
            d * 4,
            config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, config.layers, norm=nn.LayerNorm(d))
        self.output = nn.Linear(d, 7)

    def forward(self, vision, text, state):
        visual = (self.vision_proj(vision) + self.camera_embedding).flatten(1, 2)
        context = torch.stack([self.text_proj(text), self.state_proj(state)], dim=1)
        memory = torch.cat([visual, context + self.modality_embedding], dim=1)
        queries = self.action_queries.expand(state.shape[0], -1, -1)
        return self.output(self.decoder(queries, memory))


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = FrozenCLIP(config)
        self.head = ActionHead(config, self.backbone.vision_dim, self.backbone.text_dim)

    def forward(self, images, state, language):
        vision, text = self.backbone(images, language)
        return self.head(vision, text, state)


def action_loss(predictions, targets, mask):
    # Compute losses in FP32 and exclude padded timesteps from both terms.
    predictions, targets = predictions.float(), targets.float()
    mask = mask.float()
    count = mask.sum().clamp_min(1)
    motion = F.smooth_l1_loss(predictions[..., :6], targets[..., :6], reduction="none")
    motion = (motion.mean(-1) * mask).sum() / count
    gripper = F.binary_cross_entropy_with_logits(
        predictions[..., 6], targets[..., 6], reduction="none"
    )
    gripper = (gripper * mask).sum() / count
    return motion + gripper, {"motion": motion.detach(), "gripper": gripper.detach()}
