"""Shared output handling for closed-loop evaluation."""

from pathlib import Path


def prepare_eval_output(output, checkpoint_step):
    """Append the actual checkpoint step, without repeating an existing suffix."""
    out = Path(output)
    suffix = f"_{checkpoint_step}"
    if not out.name.endswith(suffix):
        out = out.with_name(out.name + suffix)
    if any((out / name).exists() for name in ("episodes.jsonl", "results.json")):
        raise FileExistsError(f"Evaluation output exists: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation output: {out.resolve()} (checkpoint step {checkpoint_step})", flush=True)
    return out
