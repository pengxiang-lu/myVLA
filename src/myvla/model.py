import hashlib

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPConfig, CLIPModel, CLIPTokenizerFast


class FrozenCLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backend = config.backend
        self.use_tokens = config.fusion_layers > 0
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
        texts, tokens = [], []
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
                encoded = self.clip.text_model(**inputs)
                self.text_cache[language] = (
                    encoded.pooler_output[0].float().cpu(),
                    encoded.last_hidden_state[0].float().cpu() if self.use_tokens else None,
                )
            pooled, sequence = self.text_cache[language]
            texts.append(pooled)
            if self.use_tokens:
                tokens.append(sequence)
        pooled = torch.stack(texts).to(device)
        if not self.use_tokens:
            return vision.float(), pooled
        lengths = torch.tensor([len(t) for t in tokens], device=device)
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True).to(device)
        padding = torch.arange(tokens.shape[1], device=device)[None] >= lengths[:, None]
        return vision.float(), pooled, tokens, padding


class VisionLanguageFusion(nn.Module):
    """Bidirectional token interaction with zero-gated residual initialization."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.visual_norm = nn.LayerNorm(d)
        self.text_norm = nn.LayerNorm(d)
        self.visual_attention = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.text_attention = nn.MultiheadAttention(d, heads, dropout, batch_first=True)
        self.visual_ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.text_ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d, d),
        )
        self.gates = nn.Parameter(torch.zeros(4))

    def forward(self, visual, text, padding):
        v, t = self.visual_norm(visual), self.text_norm(text)
        dv = self.visual_attention(v, t, t, key_padding_mask=padding, need_weights=False)[0]
        dt = self.text_attention(t, v, v, need_weights=False)[0]
        gates = self.gates.tanh()
        visual = visual + gates[0] * dv
        text = text + gates[1] * dt
        visual = visual + gates[2] * self.visual_ffn(visual)
        text = text + gates[3] * self.text_ffn(text)
        return visual, text


class ActionHead(nn.Module):
    def __init__(self, config, vision_dim, text_dim):
        super().__init__()
        d = config.hidden_dim
        self.vision_proj = nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, d))
        self.text_proj = nn.Linear(text_dim, d)
        self.state_proj = nn.Sequential(nn.Linear(config.state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.camera_embedding = nn.Parameter(torch.randn(1, config.cameras, 1, d) * 0.02)
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
        self.output = nn.Linear(d, config.action_dim)
        self.fusion = nn.ModuleList(
            [
                VisionLanguageFusion(d, config.heads, config.dropout)
                for _ in range(config.fusion_layers)
            ]
        )

    def forward(self, vision, text, state, tokens=None, padding=None):
        visual = (self.vision_proj(vision) + self.camera_embedding).flatten(1, 2)
        if self.fusion:
            token_features = self.text_proj(tokens)
            original_tokens = token_features
            for layer in self.fusion:
                visual, token_features = layer(visual, token_features, padding)
            # Add the EOS interaction residual to the original pooled instruction.
            eos = (~padding).sum(1) - 1
            indices = torch.arange(len(eos), device=eos.device)
            text_summary = self.text_proj(text) + (token_features - original_tokens)[indices, eos]
        else:
            text_summary = self.text_proj(text)
        context = torch.stack([text_summary, self.state_proj(state)], dim=1)
        memory = torch.cat([visual, context + self.modality_embedding], dim=1)
        queries = self.action_queries.expand(state.shape[0], -1, -1)
        return self.output(self.decoder(queries, memory))


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = FrozenCLIP(config)
        self.head = ActionHead(config, self.backbone.vision_dim, self.backbone.text_dim)

    def forward(self, images, state, language):
        features = self.backbone(images, language)
        return self.head(features[0], features[1], state, *features[2:])


def action_loss(predictions, targets, mask):
    # Compute losses in FP32 and exclude padded timesteps from both terms.
    predictions, targets = predictions.float(), targets.float()
    mask = mask.float()
    count = mask.sum().clamp_min(1)
    if predictions.shape[-1] == 4:
        # MetaWorld gripper is a continuous control, just like xyz.
        loss = F.smooth_l1_loss(predictions, targets, reduction="none").mean(-1)
        loss = (loss * mask).sum() / count
        return loss, {"continuous": loss.detach()}
    motion = F.smooth_l1_loss(predictions[..., :6], targets[..., :6], reduction="none")
    motion = (motion.mean(-1) * mask).sum() / count
    gripper = F.binary_cross_entropy_with_logits(
        predictions[..., 6], targets[..., 6], reduction="none"
    )
    gripper = (gripper * mask).sum() / count
    return motion + gripper, {"motion": motion.detach(), "gripper": gripper.detach()}
