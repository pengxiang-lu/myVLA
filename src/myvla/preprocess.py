"""The same image, state and action conventions are used offline and online."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]


def augment_images(images, shift=0, color_jitter=0.0):
    """Training-only shifts and color jitter on normalized images, using torch RNG.

    Run in the training process after device transfer, so checkpoint RNG restoration
    also restores augmentation without depending on DataLoader worker prefetch.
    No flips/rotations: those would change the spatial meaning of language/actions.
    """
    if shift == 0 and color_jitter == 0:
        return images
    shape = images.shape
    x = images.flatten(0, 1)
    n, channels, height, width = x.shape
    if shift:
        padded = F.pad(x, (shift, shift, shift, shift), mode="replicate")
        offsets = torch.randint(2 * shift + 1, (n, 2), device=x.device)
        rows = torch.arange(height, device=x.device)[None, :] + offsets[:, :1]
        cols = torch.arange(width, device=x.device)[None, :] + offsets[:, 1:]
        x = padded.gather(2, rows[:, None, :, None].expand(n, channels, height, width + 2 * shift))
        x = x.gather(3, cols[:, None, None, :].expand(n, channels, height, width))
    if color_jitter:
        mean, std = CLIP_MEAN.to(x), CLIP_STD.to(x)
        x = x * std + mean
        factors = 1 + color_jitter * (2 * torch.rand(n, 2, 1, 1, 1, device=x.device) - 1)
        center = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - center) * factors[:, 0] + center) * factors[:, 1]
        x = (x.clamp(0, 1) - mean) / std
    return x.reshape(shape)


def prepare_image(image, size, flip_vertical):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected RGB uint8 HWC, got {image.shape}/{image.dtype}")
    if flip_vertical:
        image = image[::-1]
    # LIBERO cameras are square. Explicit resize avoids hidden center-crop differences.
    image = Image.fromarray(np.ascontiguousarray(image)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
    return (tensor - CLIP_MEAN) / CLIP_STD


def quat_to_axis_angle(quat):
    """Match robosuite 1.4 quat2axisangle (xyzw), including its sign convention."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1, 1)
    den = np.sqrt(1 - quat[3] ** 2)
    if np.isclose(den, 0, atol=1e-12, rtol=0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2 * np.arccos(quat[3]) / den).astype(np.float32)


def state_from_observation(obs):
    return np.concatenate(
        [
            obs["robot0_eef_pos"],
            quat_to_axis_angle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)


def normalize_state(state, stats):
    return (np.asarray(state, np.float32) - stats["state_mean"]) / stats["state_std"]


def normalize_actions(actions, stats):
    result = np.asarray(actions, np.float32).copy()
    if len(stats["action_mean"]) == 4:
        return (result - stats["action_mean"]) / stats["action_std"]
    result[..., :6] = (result[..., :6] - stats["action_mean"]) / stats["action_std"]
    # LIBERO Panda convention: negative=open, positive=close. Do not invert at rollout.
    result[..., 6] = (result[..., 6] > 0).astype(np.float32)
    return result


def decode_actions(predictions, stats):
    result = np.asarray(predictions, np.float32).copy()
    if len(stats["action_mean"]) == 4:
        result = result * stats["action_std"] + stats["action_mean"]
    else:
        result[..., :6] = result[..., :6] * stats["action_std"] + stats["action_mean"]
        result[..., 6] = np.where(result[..., 6] >= 0, 1, -1)  # logit threshold
    if not np.isfinite(result).all():
        raise ValueError("Non-finite actions; rollout stopped")
    return np.clip(result, -1, 1)
