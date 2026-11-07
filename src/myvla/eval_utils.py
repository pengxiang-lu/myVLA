"""Durable, resumable output handling shared by closed-loop evaluators."""

import hashlib
import json
from pathlib import Path

import yaml


def atomic_write(path, text):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def prepare_eval_output(output, checkpoint_step):
    """Append the actual checkpoint step; existing resumable runs are validated later."""
    out = Path(output)
    suffix = f"_{checkpoint_step}"
    if not out.name.endswith(suffix):
        out = out.with_name(out.name + suffix)
    if any((out / name).exists() for name in ("episodes.jsonl", "results.json")):
        if not (out / "progress.json").exists():
            raise FileExistsError(f"Evaluation output exists: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation output: {out.resolve()} (checkpoint step {checkpoint_step})", flush=True)
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
        # The exact serialized checkpoint config is retained, including legacy fields.
        metadata["config"] = config
        metadata["task_names"] = self.task_names
        metadata_path = self.out / "evaluation.json"
        config_path = self.out / "config.yaml"
        progress_path = self.out / "progress.json"
        if metadata_path.exists():
            if json.loads(metadata_path.read_text()) != metadata:
                raise ValueError(
                    "Evaluation checkpoint or settings differ; use a new output directory"
                )
            if not config_path.exists() or yaml.safe_load(config_path.read_text()) != config:
                raise ValueError("Evaluation config.yaml is missing or differs from the checkpoint")
        else:
            if progress_path.exists():
                raise ValueError("Cannot resume progress.json without matching evaluation.json")
            atomic_write(config_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            atomic_write(metadata_path, json.dumps(metadata, indent=2))
        if progress_path.exists():
            self.progress = json.loads(progress_path.read_text())
            self._validate()
        else:
            self.progress = {}
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
        atomic_write(path, "".join(json.dumps(r) + "\n" for r in records.values()))

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
        atomic_write(self.out / "results.json", json.dumps(result, indent=2))
        return result
