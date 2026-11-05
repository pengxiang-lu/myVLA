"""Import the official LeRobot MT50 image/parquet dataset without installing LeRobot."""

import io
import json
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def task_definitions():
    return json.loads(Path(__file__).with_name("metaworld_tasks.json").read_text())


def task_language(name):
    return task_definitions()["descriptions"][name]


def parquet_module():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("MT50 import requires `uv sync --extra mt50`") from exc
    return pq


def _scalar(value):
    return np.asarray(value).reshape(-1).item()


def _image(value, source, size):
    if not isinstance(value, dict):
        raise ValueError("Expected LeRobot embedded image {bytes, path}, not video references")
    if value.get("bytes") is not None:
        handle = io.BytesIO(value["bytes"])
    elif value.get("path"):
        handle = (source / value["path"]).resolve()
        if not handle.is_relative_to(source.resolve()):
            raise ValueError("Image path escapes the source dataset")
    else:
        raise ValueError("Missing RGB image bytes/path")
    with Image.open(handle) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image, got {image.mode}")
        return np.asarray(image.resize((size, size), Image.Resampling.BICUBIC))


def convert_metaworld(source, output, task_ids=None, max_episodes=None, image_size=224):
    """Stream parquet rows to per-task HDF5; publish only after successful validation.

    --max-episodes / --tasks explicitly produce a development subset. Full imports
    require all episodes/frames in info.json and all 50 task IDs. Original images are
    already camera-corrected by LeRobot; no additional flip/crop is applied here.
    """
    pq = parquet_module()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Destination already exists: {output}")
    if image_size < 1 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("image_size and max_episodes must be positive")
    definitions = task_definitions()
    by_id = {i: name for name, i in definitions["task_ids"].items()}
    selected = set(by_id if task_ids is None else task_ids)
    if not selected or not selected <= by_id.keys():
        raise ValueError("Task IDs must be unique integers in [0,49]")
    if task_ids is not None and len(selected) != len(task_ids):
        raise ValueError("Duplicate task IDs")
    info = json.loads((source / "meta/info.json").read_text())
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [4]:
            raise ValueError(f"Expected {key} shape [4]")
    if info["features"]["observation.image"]["dtype"] != "image":
        raise ValueError("Only the official embedded RGB image dataset is supported")
    files = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError("No LeRobot data/chunk-*/*.parquet files")
    episode_meta = sorted((source / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_meta:
        raise FileNotFoundError("Missing LeRobot v3 meta/episodes parquet index")
    expected = {}
    for path in episode_meta:
        for row in pq.read_table(path, columns=["episode_index", "length"]).to_pylist():
            expected[int(row["episode_index"])] = int(row["length"])
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, lengths, identities = Counter(), {}, {}
    clipped_action_frames = 0
    full = task_ids is None and max_episodes is None
    source_files = []
    with tempfile.TemporaryDirectory(prefix=".mt50-import-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        staging.mkdir()
        handles = {}
        try:
            for path in files:
                stat = path.stat()
                source_files.append(
                    {
                        "path": str(path.relative_to(source)),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
                columns = [
                    "episode_index",
                    "frame_index",
                    "task_id",
                    "action",
                    "observation.state",
                    "observation.image",
                ]
                for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
                    for row in batch.to_pylist():
                        task_id = int(_scalar(row["task_id"]))
                        episode_id = int(_scalar(row["episode_index"]))
                        if task_id not in by_id:
                            raise ValueError(f"Invalid task_id: {task_id}")
                        if task_id not in selected:
                            continue
                        if (
                            episode_id not in lengths
                            and max_episodes is not None
                            and len(lengths) >= max_episodes
                        ):
                            continue
                        name = by_id[task_id]
                        if episode_id in identities and identities[episode_id] != task_id:
                            raise ValueError("Task identity changes inside a trajectory")
                        identities[episode_id] = task_id
                        t = lengths.get(episode_id, 0)
                        if int(_scalar(row["frame_index"])) != t:
                            raise ValueError("Non-contiguous/duplicate trajectory frame indices")
                        state = np.asarray(row["observation.state"], np.float32)
                        action = np.asarray(row["action"], np.float32)
                        if (
                            state.shape != (4,)
                            or action.shape != (4,)
                            or not np.isfinite(state).all()
                            or not np.isfinite(action).all()
                        ):
                            raise ValueError("Invalid MT50 state/action")
                        # Scripted xyz targets can exceed the action space. MetaWorld
                        # clips xyz internally; learn the effective bounded control.
                        clipped_action_frames += int(np.abs(action).max() > 1)
                        action = np.clip(action, -1, 1)
                        rgb = _image(row["observation.image"], source, image_size)
                        if task_id not in handles:
                            f = h5py.File(staging / f"{name}_demo.hdf5", "w")
                            handles[task_id] = f
                            f.attrs["dataset"] = "metaworld_mt50"
                            f.attrs["task_name"] = name
                            f.attrs["task_id"] = task_id
                            data = f.create_group("data")
                            data.attrs["problem_info"] = json.dumps(
                                {"language_instruction": task_language(name)}
                            )
                        data = handles[task_id]["data"]
                        demo = f"demo_{episode_id}"
                        if t == 0:
                            group = data.create_group(demo)
                            for key, shape, dtype in [
                                ("actions", (4,), "f4"),
                                ("obs/state", (4,), "f4"),
                                ("obs/rgb", (image_size, image_size, 3), "u1"),
                            ]:
                                group.create_dataset(
                                    key,
                                    shape=(0, *shape),
                                    maxshape=(None, *shape),
                                    chunks=(1, *shape),
                                    dtype=dtype,
                                    compression="gzip",
                                    compression_opts=1,
                                )
                            counts[task_id] += 1
                        for key, value in [
                            ("actions", action),
                            ("obs/state", state),
                            ("obs/rgb", rgb),
                        ]:
                            dataset = data[demo][key]
                            dataset.resize(t + 1, axis=0)
                            dataset[t] = value
                        lengths[episode_id] = t + 1
                    if (
                        max_episodes is not None
                        and len(lengths) >= max_episodes
                        and all(expected.get(ep) == n for ep, n in lengths.items())
                    ):
                        break
                print(
                    json.dumps(
                        {
                            "file": str(path),
                            "episodes": len(lengths),
                            "frames": sum(lengths.values()),
                        }
                    ),
                    flush=True,
                )
                if (
                    max_episodes is not None
                    and len(lengths) >= max_episodes
                    and all(expected.get(ep) == n for ep, n in lengths.items())
                ):
                    break
            if not lengths:
                raise ValueError("No selected episodes found")
            if full and (
                len(lengths) != info["total_episodes"]
                or sum(lengths.values()) != info["total_frames"]
                or set(counts) != selected
            ):
                raise ValueError(
                    "Incomplete MT50 snapshot: episode/frame/task counts differ from metadata"
                )
            if min(counts.values()) < 2:
                raise ValueError(
                    "Need >=2 complete demonstrations per selected task for train/val split"
                )
            if max_episodes is None and set(counts) != selected:
                raise ValueError("Missing requested task IDs in the source snapshot")
            # Reject truncated episodes even for explicit development subsets.
            if any(expected.get(ep) != length for ep, length in lengths.items()):
                raise ValueError("Incomplete trajectory: length differs from episode metadata")
        finally:
            for f in handles.values():
                f.close()
        report = {
            "source": str(source.resolve()),
            "source_info": info,
            "source_files": source_files,
            "image_size": image_size,
            "actions_clipped_to_bounds": True,
            "clipped_action_frames": clipped_action_frames,
            "episodes": len(lengths),
            "frames": sum(lengths.values()),
            "full_mt50": full,
            "task_episodes": {by_id[i]: n for i, n in sorted(counts.items())},
        }
        (staging / "conversion.json").write_text(json.dumps(report, indent=2))
        staging.rename(output)
    return report
