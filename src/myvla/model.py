import hashlib
import math
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, CLIPConfig, CLIPModel, CLIPTokenizerFast, SiglipModel

from .preprocess import CLIP_MEAN, CLIP_STD


class FrozenCLIP(nn.Module):
    """CLIP/SigLIP adapter; keep the legacy name and `clip` checkpoint namespace."""
    def __init__(self, config):
        super().__init__()
        self.backend = config.backend
        self.is_siglip = self.backend == "siglip"
        self.use_tokens = config.fusion_layers > 0
        self.train_vision_encoder = config.train_vision_encoder
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
            model_class = SiglipModel if self.is_siglip else CLIPModel
            self.clip = model_class.from_pretrained(
                config.backbone, revision=config.revision, attn_implementation="sdpa"
            )
            # Store the resolved commit rather than a moving branch in checkpoints.
            config.revision = self.clip.config._commit_hash or config.revision
            tokenizer_class = AutoTokenizer if self.is_siglip else CLIPTokenizerFast
            self.tokenizer = tokenizer_class.from_pretrained(
                config.backbone, revision=config.revision
            )
        self.clip.requires_grad_(False)
        self.clip.vision_model.requires_grad_(self.train_vision_encoder)
        self.clip.eval()
        self.vision_dim = self.clip.config.vision_config.hidden_size
        self.text_dim = self.clip.config.text_config.hidden_size
        self.text_cache = {}

    def train(self, mode=True):
        super().train(False)
        self.clip.vision_model.train(mode and self.train_vision_encoder)
        return self

    def tokenize(self, language, device):
        if self.tokenizer is None:
            digest = hashlib.sha256(language.encode()).digest()[:16]
            ids = [1] + [3 + b % 125 for b in digest] + [2]
            return {"input_ids": torch.tensor([ids], device=device)}
        if self.is_siglip:
            inputs = self.tokenizer(
                language, return_tensors="pt", truncation=True, padding="max_length",
                max_length=self.clip.config.text_config.max_position_embeddings,
            )
            # SigLIP was pretrained without masking its fixed-length EOS padding.
            return {"input_ids": inputs["input_ids"].to(device)}
        return self.tokenizer(
            language, return_tensors="pt", truncation=True, max_length=77
        ).to(device)

    def encode_vision(self, images):
        batch, cameras = images.shape[:2]
        if self.is_siglip:
            # All data/rollout/augmentation APIs use CLIP-normalized images.
            # Convert once here to SigLIP's RGB mean=std=0.5 convention.
            images = ((images * CLIP_STD.to(images) + CLIP_MEAN.to(images)) - 0.5) / 0.5
        context = nullcontext() if self.train_vision_encoder else torch.no_grad()
        with context:
            vision = self.clip.vision_model(pixel_values=images.flatten(0, 1)).last_hidden_state
        if not self.is_siglip:
            vision = vision[:, 1:]  # CLIP has CLS; SigLIP contains only patch tokens.
        return vision.reshape(batch, cameras, -1, self.vision_dim).float()

    def forward(self, images, languages):
        vision = self.encode_vision(images)
        return (vision.float(), *self.encode_text(languages, images.device))

    @torch.no_grad()
    def encode_text(self, languages, device):
        texts, tokens = [], []
        for language in languages:
            if language not in self.text_cache:
                inputs = self.tokenize(language, device)
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
            return (pooled,)
        lengths = torch.tensor([len(t) for t in tokens], device=device)
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True).to(device)
        padding = torch.arange(tokens.shape[1], device=device)[None] >= lengths[:, None]
        return pooled, tokens, padding


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

    def encode_context(self, vision, text, state, tokens=None, padding=None):
        visual = (self.vision_proj(vision) + self.camera_embedding).flatten(1, 2)
        if self.fusion:
            token_features = self.text_proj(tokens)
            original_tokens = token_features
            for layer in self.fusion:
                visual, token_features = layer(visual, token_features, padding)
            # Last valid token: CLIP EOS or SigLIP's final fixed-length padding token.
            eos = (~padding).sum(1) - 1
            indices = torch.arange(len(eos), device=eos.device)
            text_summary = self.text_proj(text) + (token_features - original_tokens)[indices, eos]
        else:
            text_summary = self.text_proj(text)
        context = torch.stack([text_summary, self.state_proj(state)], dim=1)
        return torch.cat([visual, context + self.modality_embedding], dim=1)

    def forward(self, vision, text, state, tokens=None, padding=None):
        memory = self.encode_context(vision, text, state, tokens, padding)
        queries = self.action_queries.expand(state.shape[0], -1, -1)
        return self.output(self.decoder(queries, memory))


class FlowMatchingActionHead(ActionHead):
    """Conditional linear noise-to-action flow with a Transformer velocity field."""

    def __init__(self, config, vision_dim, text_dim):
        super().__init__(config, vision_dim, text_dim)
        d = config.hidden_dim
        self.action_dim = config.action_dim
        self.flow_steps = config.flow_steps
        self.action_proj = nn.Linear(config.action_dim, d)
        self.time_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer(
            "time_frequencies",
            torch.exp(torch.linspace(0, -math.log(10000), (d + 1) // 2)),
        )

    def velocity(self, actions, time, memory):
        phase = time.float()[:, None] * self.time_frequencies[None] * (2 * math.pi)
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        embedding = self.time_proj(embedding[:, :self.action_queries.shape[-1]])
        queries = self.action_proj(actions) + self.action_queries + embedding[:, None]
        return self.output(self.decoder(queries, memory))

    def loss(self, vision, text, state, targets, mask, tokens=None, padding=None):
        memory = self.encode_context(vision, text, state, tokens, padding)
        targets = targets.float().clone()
        if self.action_dim == 7:
            # Dataset stores LIBERO gripper in {0,1}; flow models signed {-1,+1}.
            targets[..., 6] = targets[..., 6] * 2 - 1
        valid = mask.bool().unsqueeze(-1)
        targets = targets.masked_fill(~valid, 0)
        noise = torch.randn_like(targets).masked_fill(~valid, 0)
        time = torch.rand(len(targets), device=targets.device)
        noisy = torch.lerp(noise, targets, time[:, None, None])
        prediction = self.velocity(noisy, time, memory)
        error = (prediction.float() - (targets - noise)).square().mean(-1)
        loss = error.masked_fill(~mask.bool(), 0).sum() / mask.sum().clamp_min(1)
        return loss, {"flow": loss.detach()}

    def sample(self, memory, noise=None):
        shape = (memory.shape[0], self.action_queries.shape[1], self.action_dim)
        actions = (
            torch.randn(shape, device=memory.device, dtype=torch.float32)
            if noise is None else noise.float().clone()
        )
        for step in range(self.flow_steps):
            time = actions.new_full((len(actions),), step / self.flow_steps)
            actions = actions + self.velocity(actions, time, memory).float() / self.flow_steps
        return actions

    def forward(self, vision, text, state, tokens=None, padding=None):
        memory = self.encode_context(vision, text, state, tokens, padding)
        return self.sample(memory)


class DiffusionActionHead(ActionHead):
    """Conditional epsilon prediction with a cosine DDPM schedule and DDIM (eta=0)."""

    def __init__(self, config, vision_dim, text_dim):
        super().__init__(config, vision_dim, text_dim)
        self.action_dim = config.action_dim
        self.train_steps = config.diffusion_train_steps
        self.inference_steps = config.diffusion_inference_steps
        d = config.hidden_dim
        self.action_proj = nn.Linear(config.action_dim, d)
        self.time_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer(
            "time_frequencies", torch.exp(torch.linspace(0, -math.log(10000), (d + 1) // 2))
        )
        # Compute the schedule in float64, store FP32 for training and sampling.
        times = torch.linspace(0, 1, self.train_steps + 1, dtype=torch.float64)
        cumulative = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        self.register_buffer("alpha_bar", (1 - betas).cumprod(0).float())

    def predict_noise(self, actions, timesteps, memory):
        phase = timesteps.float()[:, None] * self.time_frequencies[None]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        embedding = self.time_proj(embedding[:, :self.action_queries.shape[-1]])
        queries = self.action_proj(actions) + self.action_queries + embedding[:, None]
        return self.output(self.decoder(queries, memory))

    def loss(self, vision, text, state, targets, mask, tokens=None, padding=None):
        memory = self.encode_context(vision, text, state, tokens, padding)
        targets = targets.float().clone()
        if self.action_dim == 7:
            targets[..., 6] = targets[..., 6] * 2 - 1
        valid = mask.bool().unsqueeze(-1)
        targets = targets.masked_fill(~valid, 0)
        noise = torch.randn_like(targets).masked_fill(~valid, 0)
        timesteps = torch.randint(self.train_steps, (len(targets),), device=targets.device)
        alpha = self.alpha_bar[timesteps, None, None]
        noisy = alpha.sqrt() * targets + (1 - alpha).sqrt() * noise
        prediction = self.predict_noise(noisy, timesteps, memory)
        error = (prediction.float() - noise).square().mean(-1)
        loss = error.masked_fill(~mask.bool(), 0).sum() / mask.sum().clamp_min(1)
        return loss, {"diffusion": loss.detach()}

    def sample(self, memory, noise=None):
        shape = (memory.shape[0], self.action_queries.shape[1], self.action_dim)
        actions = (torch.randn(shape, device=memory.device, dtype=torch.float32)
                   if noise is None else noise.float().clone())
        steps = torch.linspace(self.train_steps - 1, 0, self.inference_steps).round().long()
        for i, step in enumerate(steps.tolist()):
            timestep = torch.full((len(actions),), step, device=memory.device, dtype=torch.long)
            epsilon = self.predict_noise(actions, timestep, memory).float()
            alpha = self.alpha_bar[step]
            clean = (actions - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
            # Actions use z-score normalization, so do not clip clean predictions to [-1, 1].
            previous = (self.alpha_bar[steps[i + 1].item()]
                        if i + 1 < len(steps) else alpha.new_ones(()))
            actions = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
        return actions

    def forward(self, vision, text, state, tokens=None, padding=None):
        return self.sample(self.encode_context(vision, text, state, tokens, padding))


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = FrozenCLIP(config)
        head = {"regression": ActionHead, "flow_matching": FlowMatchingActionHead,
                "diffusion": DiffusionActionHead}[config.action_head]
        self.head = head(config, self.backbone.vision_dim, self.backbone.text_dim)

    def forward(self, images, state, language):
        features = self.backbone(images, language)
        return self.head(features[0], features[1], state, *features[2:])

    def compute_loss(self, images, state, language, targets, mask):
        if isinstance(self.head, (FlowMatchingActionHead, DiffusionActionHead)):
            features = self.backbone(images, language)
            return self.head.loss(features[0], features[1], state, targets, mask, *features[2:])
        return action_loss(self(images, state, language), targets, mask)


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
