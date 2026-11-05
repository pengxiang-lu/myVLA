"""Isolate upstream source/assets and avoid its interactive ~/.libero initialization."""

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def setup_libero(root=None):
    root = root or os.environ.get("LIBERO_ROOT")
    if root is None:
        candidate = Path("third_party/LIBERO")
        if candidate.exists():
            root = candidate
    if root is not None:
        root = Path(root).expanduser().resolve()
        if not (root / "libero/libero/__init__.py").is_file():
            raise FileNotFoundError(f"Invalid LIBERO source directory: {root}")
        sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Set LIBERO_ROOT to a LIBERO source checkout; see README")
    package_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    if not (package_root / "assets").is_dir():
        raise FileNotFoundError(f"LIBERO assets missing: {package_root}")
    source_tag = hashlib.sha256(str(package_root).encode()).hexdigest()[:12]
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", f".libero/{source_tag}")).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        dataset_dir = Path("data").resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "benchmark_root": str(package_root),
                    "bddl_files": str(package_root / "bddl_files"),
                    "init_states": str(package_root / "init_files"),
                    "assets": str(package_root / "assets"),
                    "datasets": str(dataset_dir),
                }
            )
        )
    try:
        revision = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        revision = ""
    return {
        "source_root": str(package_root.parent.parent),
        "git_revision": revision or None,
        "config_path": str(config_path),
    }


def load_init_states(suite, task_id):
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # Upstream initial states contain NumPy arrays; explicitly allow just those types.
    # PyTorch >=2.6 otherwise rejects them with its weights_only default.
    allowed = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("float32")),
    ]
    with torch.serialization.safe_globals(allowed):
        states = torch.load(path, map_location="cpu", weights_only=True)
    return states, hashlib.sha256(path.read_bytes()).hexdigest()
