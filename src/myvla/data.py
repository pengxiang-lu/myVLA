import bisect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import normalize_actions, normalize_state, prepare_image


@dataclass(frozen=True)
class Episode:
    path: str
    demo: str
    length: int
    language: str
    split: str
    file_size: int
    file_mtime_ns: int


def read_language(f):
    raw = f["data"].attrs.get("problem_info")
    if raw is None:
        raise ValueError(f"{f.filename}: missing data.attrs['problem_info']")
    if isinstance(raw, bytes):
        raw = raw.decode()
    language = json.loads(raw)["language_instruction"]
    if isinstance(language, list):
        language = "".join(language)
    if not isinstance(language, str) or not language.strip():
        raise ValueError(f"{f.filename}: empty or invalid task language")
    return language.strip()


def raw_state(obs, index, dataset="libero"):
    if dataset == "metaworld":
        return obs["state"][index].astype(np.float32)
    if "ee_states" in obs:
        ee = obs["ee_states"][index]
    else:
        ee = np.concatenate([obs["ee_pos"][index], obs["ee_ori"][index]], axis=-1)
    return np.concatenate([ee, obs["gripper_states"][index]], axis=-1).astype(np.float32)


def scan_episodes(config, seed):
    files = sorted(Path(config.directory).glob(config.pattern))
    if not files:
        raise FileNotFoundError(f"No HDF5 files: {config.directory}/{config.pattern}")
    episodes = []
    for path in files:
        stat = path.stat()
        with h5py.File(path, "r") as f:
            meta = config.dataset == "metaworld"
            if meta:
                if f.attrs.get("dataset") != "metaworld_mt50":
                    raise ValueError(f"{path}: not a converted MT50 HDF5 file")
            language = read_language(f)
            demos = sorted(f["data"], key=lambda s: int(s.split("_")[-1]))
            if not demos:
                raise ValueError(f"{path}: no demonstrations")
            for demo in demos:
                group = f[f"data/{demo}"]
                n = len(group["actions"])
                action_dim = 4 if meta else 7
                if n < 1 or group["actions"].shape != (n, action_dim):
                    raise ValueError(f"{path}/{demo}: actions must have shape (T,{action_dim})")
                obs = group["obs"]
                image_keys = ("rgb",) if meta else ("agentview_rgb", "eye_in_hand_rgb")
                for key in image_keys + (("state",) if meta else ("gripper_states",)):
                    if len(obs[key]) != n:
                        raise ValueError(f"{path}/{demo}: {key} length mismatch")
                for key in image_keys:
                    if obs[key].ndim != 4 or obs[key].shape[-1] != 3:
                        raise ValueError(f"{path}/{demo}: invalid RGB schema for {key}")
                    if obs[key].dtype != np.uint8:
                        raise ValueError(f"{path}/{demo}: images must be uint8")
                state_key = "state" if meta else "ee_states" if "ee_states" in obs else "ee_pos"
                state_dim = 4 if meta else 8
                if len(obs[state_key]) != n or raw_state(obs, 0, config.dataset).shape != (
                    state_dim,
                ):
                    raise ValueError(
                        f"{path}/{demo}: expected {state_dim}D state aligned with actions"
                    )
                episodes.append(
                    Episode(
                        str(path.resolve()),
                        demo,
                        n,
                        language,
                        "train",
                        stat.st_size,
                        stat.st_mtime_ns,
                    )
                )
    return episodes


def compute_stats(episodes, dataset="libero"):
    """Streaming sums over all training trajectories; no image caching."""
    total = 0
    state_dim, continuous_dim = (4, 4) if dataset == "metaworld" else (8, 6)
    sums = np.zeros(state_dim + continuous_dim, np.float64)
    squares = np.zeros_like(sums)
    for episode in episodes:
        if episode.split != "train":
            continue
        with h5py.File(episode.path, "r") as f:
            group = f[f"data/{episode.demo}"]
            for start in range(0, episode.length, 4096):
                sl = slice(start, start + 4096)
                actions = group["actions"][sl].astype(np.float64)
                state = raw_state(group["obs"], sl, dataset).astype(np.float64)
                values = np.concatenate([state, actions[:, :continuous_dim]], axis=-1)
                if not np.isfinite(values).all() or not np.isfinite(actions).all():
                    raise ValueError(f"Non-finite state/action: {episode.path}/{episode.demo}")
                if np.abs(actions).max() > 1.0001:
                    raise ValueError(f"Expected {dataset} actions within [-1,1]")
                total += len(values)
                sums += values.sum(0)
                squares += np.square(values).sum(0)
    if not total:
        raise ValueError("No training frames")
    mean = sums / total
    std = np.sqrt(np.maximum(squares / total - mean**2, 0)).clip(1e-3)
    return {
        "state_mean": mean[:state_dim].astype(np.float32),
        "state_std": std[:state_dim].astype(np.float32),
        "action_mean": mean[state_dim:].astype(np.float32),
        "action_std": std[state_dim:].astype(np.float32),
    }


def manifest(episodes):
    return [asdict(e) for e in episodes]


class RobotDataset(Dataset):
    def __init__(self, episodes, split, config, model_config, stats):
        self.episodes = [e for e in episodes if e.split == split]
        self.ends = np.cumsum([e.length for e in self.episodes]).tolist()
        self.config, self.model_config, self.stats = config, model_config, stats
        if not self.ends:
            raise ValueError(f"No {split} trajectories")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = bisect.bisect_right(self.ends, index)
        ep = self.episodes[i]
        t = index - (self.ends[i - 1] if i else 0)
        k = self.model_config.chunk_size
        valid = min(k, ep.length - t)
        # Per-sample handles are fork/spawn-safe and bounded across many task files.
        with h5py.File(ep.path, "r") as f:
            group = f[f"data/{ep.demo}"]
            obs = group["obs"]
            images = torch.stack(
                [
                    prepare_image(
                        obs[key][t], self.model_config.image_size, self.config.flip_vertical
                    )
                    for key in (
                        ("rgb",)
                        if self.config.dataset == "metaworld"
                        else ("agentview_rgb", "eye_in_hand_rgb")
                    )
                ]
            )
            state = normalize_state(raw_state(obs, t, self.config.dataset), self.stats)
            actions = np.zeros((k, self.model_config.action_dim), np.float32)
            actions[:valid] = normalize_actions(group["actions"][t : t + valid], self.stats)
        return {
            "images": images,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "mask": torch.arange(k) < valid,
            "language": ep.language,
        }


# Keep the original public name for existing scripts and checkpoints' companion tooling.
LiberoDataset = RobotDataset
