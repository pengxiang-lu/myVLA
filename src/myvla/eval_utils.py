"""Durable, resumable output handling shared by closed-loop evaluators."""

import hashlib
import json
import re
from pathlib import Path

import yaml


def atomic_write(path, text):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def prepare_eval_output(output, checkpoint_step, seed=42):
    """Group evaluations by experiment, checkpoint step and evaluation seed."""
    out = Path(output)
    seed_name = f"seed{seed}"
    if re.fullmatch(r"seed-?\d+", out.name):
        if out.name != seed_name:
            raise ValueError(f"Output seed directory must be {seed_name}, got {out.name}")
        if out.parent.name != f"ckpt{checkpoint_step}":
            raise ValueError(f"Output checkpoint directory must be ckpt{checkpoint_step}")
        out = out.parent
    name = f"ckpt{checkpoint_step}"
    if re.fullmatch(r"ckpt_?\d+", out.name):
        if out.name != name:
            raise ValueError(f"Output checkpoint directory must be {name}, got {out.name}")
    else:
        out = out / name
    out = out / seed_name
    if any((out / name).exists() for name in ("episodes.jsonl", "log.json", "results.json")):
        if not (out / "progress.json").exists():
            raise FileExistsError(f"Evaluation output exists: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"Evaluation output: {out.resolve()} (checkpoint step {checkpoint_step}, seed {seed})",
        flush=True,
    )
    return out


class EvaluationProgress:
    def __init__(self, out, checkpoint, config, metadata, task_names, episodes):
        self.out = Path(out)
        self.episodes = episodes
        if episodes < 1:
            raise ValueError("episodes must be positive")
        self.task_names = list(task_names)
        if not self.task_names or len(set(self.task_names)) != len(self.task_names):
            raise ValueError("Evaluation task names must be nonempty and unique")
        digest = hashlib.sha256()
        with Path(checkpoint).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata = dict(
            metadata, checkpoint_sha256=digest.hexdigest(), format_version=2,
            episodes_per_task=episodes,
        )
        metadata["task_names"] = self.task_names
        # Keep resume identity in the existing detailed log, without a separate config file.
        self.header = dict(type="evaluation_metadata", config=config, evaluation=metadata)
        log_path = self.out / "episodes.jsonl"
        log_text = log_path.read_text() if log_path.exists() else ""
        try:
            first = json.loads(log_text.splitlines()[0]) if log_text else None
        except (ValueError, IndexError):
            first = None
        config_path = self.out / "config.yaml"
        legacy_path = self.out / "evaluation.json"
        has_header = isinstance(first, dict) and first.get("type") == "evaluation_metadata"
        if has_header:
            stored = first
        elif legacy_path.exists():
            legacy = json.loads(legacy_path.read_text())
            legacy_config = legacy.pop("config")
            stored = dict(type="evaluation_metadata", config=legacy_config, evaluation=legacy)
        elif config_path.exists():
            legacy_config = yaml.safe_load(config_path.read_text())
            if not isinstance(legacy_config, dict) or "evaluation" not in legacy_config:
                raise ValueError("Legacy evaluation config has no resume metadata")
            legacy = legacy_config.pop("evaluation")
            stored = dict(type="evaluation_metadata", config=legacy_config, evaluation=legacy)
        else:
            stored = None
        if stored is not None and stored != self.header:
            raise ValueError(
                "Evaluation checkpoint or settings differ; use a new output directory"
            )
        progress_path = self.out / "progress.json"
        if progress_path.exists():
            if stored is None:
                raise ValueError(
                    "Cannot resume progress.json without matching episode-log metadata"
                )
            self.progress = json.loads(progress_path.read_text())
            self._validate()
        else:
            self.progress = {}
        if not has_header:
            atomic_write(log_path, json.dumps(self.header) + "\n" + log_text)
        # Remove old metadata only after its replacement has been safely written.
        config_path.unlink(missing_ok=True)
        legacy_path.unlink(missing_ok=True)
        for task in self.task_names:
            self.progress.setdefault(task, {})
        self._save()
        self._reconcile_log()

    def _validate(self):
        if not isinstance(self.progress, dict) or set(self.progress) - set(self.task_names):
            raise ValueError("progress.json contains unexpected tasks")
        allowed = {str(i) for i in range(self.episodes)}
        for task, records in self.progress.items():
            if not isinstance(records, dict) or set(records) - allowed:
                raise ValueError(f"Invalid episode indices in progress.json: {task}")
            if any(type(success) is not bool for success in records.values()):
                raise ValueError(f"Episode outcomes must be booleans: {task}")

    def _save(self):
        atomic_write(self.out / "progress.json", json.dumps(self.progress, indent=2))

    def _reconcile_log(self):
        """Discard uncommitted/duplicate rows and a torn append after interruption."""
        path = self.out / "episodes.jsonl"
        if not path.exists():
            return
        records = {}
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
                task, episode = record["task"], str(record["episode"])
            except (ValueError, KeyError, TypeError):
                continue
            if (
                task in self.progress and episode in self.progress[task]
                and record.get("success") is self.progress[task][episode]
            ):
                records[task, episode] = record
        atomic_write(
            path, json.dumps(self.header) + "\n"
            + "".join(json.dumps(r) + "\n" for r in records.values()),
        )

    def done(self, task, episode):
        # A failed episode is completed too: never use its boolean value as the check.
        return str(episode) in self.progress[task]

    def task_done(self, task):
        return len(self.progress[task]) == self.episodes

    def record(self, task, episode, success, details):
        if task not in self.progress or not 0 <= episode < self.episodes:
            raise ValueError("Unknown task or episode")
        if self.done(task, episode):
            raise ValueError("Episode already completed")
        success = bool(success)
        # Append diagnostics first; progress.json is the authoritative commit.
        with (self.out / "episodes.jsonl").open("a", buffering=1) as log:
            log.write(json.dumps(dict(details, task=task, episode=episode, success=success)) + "\n")
        self.progress[task][str(episode)] = success
        self._save()

    def finish(self):
        if not all(self.task_done(task) for task in self.task_names):
            raise ValueError("Cannot summarize an incomplete evaluation")
        rates = {
            task: sum(self.progress[task].values()) / self.episodes for task in self.task_names
        }
        result = {"success_rate": rates, "total_success_rate": sum(rates.values()) / len(rates)}
        atomic_write(self.out / "log.json", json.dumps(result, indent=2))
        return result
