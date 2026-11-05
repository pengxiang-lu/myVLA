"""Evaluate myVLA: uv run evaluate.py --checkpoint runs/.../best.pt --output runs/eval.

The checkpoint determines the dataset by default. Relative paths use the project root.
"""

import argparse
import os
from pathlib import Path

from myvla.train import load_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VLA policy on LIBERO or MetaWorld MT50.",
        epilog="Relative checkpoint, output and LIBERO source paths use the project root.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        choices=["libero", "metaworld"],
        help="Optional check; defaults to the dataset recorded in the checkpoint",
    )
    parser.add_argument(
        "--suite",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO suite (default: libero_spatial)",
    )
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument(
        "--episodes", type=int, help="Episodes per task (default: LIBERO 10, MetaWorld 50)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--execute-steps", type=int, help="Actions per observation (default: LIBERO 4, MetaWorld 1)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--horizon", type=int, help="Control-step limit (default: suite-specific; MetaWorld 500)"
    )
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--libero-root", help="LIBERO source checkout directory")
    parser.add_argument(
        "--protocol", choices=["lerobot", "benchmark"], help="MetaWorld protocol (default: lerobot)"
    )
    args = parser.parse_args(argv)
    os.chdir(PROJECT_ROOT)
    # Release the temporary payload before the evaluator loads the actual policy.
    recorded = load_checkpoint(args.checkpoint)["config"].get("data", {}).get("dataset", "libero")
    if args.dataset is not None and args.dataset != recorded:
        parser.error(f"Checkpoint dataset is {recorded}, not {args.dataset}")
    if recorded not in {"libero", "metaworld"}:
        parser.error(f"Unsupported checkpoint dataset: {recorded}")
    meta = recorded == "metaworld"
    if meta and (args.suite is not None or args.libero_root is not None):
        parser.error("--suite and --libero-root apply only to LIBERO")
    if not meta and args.protocol is not None:
        parser.error("--protocol applies only to MetaWorld")
    common = {
        "checkpoint": args.checkpoint,
        "output": args.output,
        "tasks": args.tasks,
        "episodes": args.episodes if args.episodes is not None else (50 if meta else 10),
        "device_name": args.device,
        "execute_steps": (
            args.execute_steps if args.execute_steps is not None else (1 if meta else 4)
        ),
        "seed": args.seed,
        "horizon": args.horizon if args.horizon is not None else (500 if meta else None),
        "videos": args.videos,
    }
    if meta:
        from myvla.metaworld_evaluate import evaluate_metaworld

        return evaluate_metaworld(**common, protocol=args.protocol or "lerobot")
    from myvla.evaluate import evaluate

    return evaluate(**common, suite=args.suite or "libero_spatial", libero_root=args.libero_root)


if __name__ == "__main__":
    main()
