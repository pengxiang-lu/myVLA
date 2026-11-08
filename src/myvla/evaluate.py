import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

# Support both package imports and `uv run src/myvla/evaluate.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "myvla"

import numpy as np
import torch

from .eval_utils import EvaluationProgress, prepare_eval_output
from .libero_runtime import load_init_states, setup_libero
from .preprocess import decode_actions, normalize_state, prepare_image, state_from_observation
from .train import load_checkpoint, policy_from_checkpoint, resolve_device, seed_everything

HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class Controller:
    def __init__(self, policy, config, stats, device, execute_steps):
        if not 1 <= execute_steps <= config.model.chunk_size:
            raise ValueError("execute_steps must be in [1, chunk_size]")
        self.policy, self.config, self.stats = policy, config, stats
        self.device, self.execute_steps = device, execute_steps
        self.queue = []

    def reset(self):
        self.queue.clear()

    @torch.no_grad()
    def act(self, obs, language):
        if not self.queue:
            images = torch.stack(
                [
                    prepare_image(
                        obs[k], self.config.model.image_size, self.config.data.flip_vertical
                    )
                    for k in ("agentview_image", "robot0_eye_in_hand_image")
                ]
            )
            state = normalize_state(state_from_observation(obs), self.stats)
            pred = self.policy(
                images[None].to(self.device),
                torch.from_numpy(state)[None].to(self.device),
                [language],
            )
            actions = decode_actions(pred[0].float().cpu().numpy(), self.stats)
            self.queue = list(actions[: self.execute_steps])
        return self.queue.pop(0)


def run_episode(env, controller, language, initial_state, horizon, settle_steps=10, on_frame=None):
    if horizon < 1 or settle_steps < 0:
        raise ValueError("horizon must be positive and settle_steps non-negative")
    controller.reset()
    env.reset()
    obs = env.set_init_state(initial_state)
    # Match common LIBERO rollout convention: settle physics with an open gripper.
    for _ in range(settle_steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    for step in range(horizon):
        if on_frame:
            on_frame(obs)
        obs, _, done, _ = env.step(controller.act(obs, language))
        if env.check_success():
            return True, step + 1
        if done:
            break
    return False, step + 1


def evaluate(
    checkpoint,
    suite,
    tasks,
    episodes,
    device_name,
    execute_steps,
    output,
    seed=42,
    horizon=None,
    videos=False,
    libero_root=None,
    weights="auto",
):
    if episodes < 1 or (horizon is not None and horizon < 1):
        raise ValueError("episodes and horizon must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    source_info = setup_libero(libero_root)
    # Lazy imports keep training independent of simulator dependencies.
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise RuntimeError("LIBERO simulator missing; run `uv sync --extra sim`") from exc
    device = resolve_device(device_name)
    seed_everything(seed)
    policy, config, stats, payload = policy_from_checkpoint(checkpoint, device, weights=weights)
    if config.data.dataset != "libero":
        raise ValueError("Use eval-metaworld for a MetaWorld checkpoint")
    if config.model.backend != "clip":
        raise ValueError("Tiny random smoke model cannot be used for benchmark reporting")
    controller = Controller(policy, config, stats, device, execute_steps)
    benchmark_suite = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    task_ids = list(range(benchmark_suite.n_tasks)) if tasks is None else tasks
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Select at least one task, without duplicate IDs")
    if any(i < 0 or i >= benchmark_suite.n_tasks for i in task_ids):
        raise ValueError("Task ID out of bounds")
    plan = []
    for task_id in task_ids:
        task = benchmark_suite.get_task(task_id)
        states, states_sha256 = load_init_states(benchmark_suite, task_id)
        if episodes > len(states):
            raise ValueError(
                f"Requested {episodes} unique episodes; only {len(states)} init states"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
        indices = rng.permutation(len(states))[:episodes].tolist()
        plan.append((task_id, task, states, states_sha256, indices))
    out = prepare_eval_output(output, payload["step"], seed)
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_step": payload["step"],
        "output_dir": str(out.resolve()),
        "weights": payload["inference_weights"],
        "suite": suite,
        "seed": seed,
        "episodes_per_task": episodes,
        "execute_steps": execute_steps,
        "horizon": horizon or HORIZONS[suite],
        "settle_steps": 10,
        "control": "relative OSC_POSE",
        "libero": source_info,
        "device": str(device),
        "videos": videos,
        "tasks": [
            {"task_id": task_id, "task": task.name, "initial_states_sha256": sha,
             "initial_state_indices": indices}
            for task_id, task, _, sha, indices in plan
        ],
    }
    result["versions"] = {
        name: version(name)
        for name in ("myvla", "torch", "transformers", "robosuite", "mujoco", "numpy")
    }
    progress = EvaluationProgress(
        out, checkpoint, payload["config"], result,
        [task.name for _, task, _, _, _ in plan], episodes,
    )
    video_dir = out / "videos"
    if videos:
        video_dir.mkdir(exist_ok=True)
    for task_id, task, states, _, indices in plan:
        if progress.task_done(task.name):
            continue
        env = OffScreenRenderEnv(
            bddl_file_name=benchmark_suite.get_task_bddl_file_path(task_id),
            camera_heights=128,
            camera_widths=128,
        )
        try:
            for episode, index in enumerate(indices):
                if progress.done(task.name, episode):
                    continue
                episode_seed = seed + task_id * 10000 + index
                seed_everything(episode_seed)
                env.seed(episode_seed)
                writer = None
                try:
                    callback = None
                    if videos:
                        import imageio.v2 as imageio

                        writer = imageio.get_writer(
                            str(video_dir / f"task{task_id:02d}_episode{episode:03d}.mp4"), fps=20
                        )

                        def callback(obs):
                            frame = obs["agentview_image"]
                            if config.data.flip_vertical:
                                frame = frame[::-1]
                            writer.append_data(np.ascontiguousarray(frame))

                    success, steps = run_episode(
                        env,
                        controller,
                        task.language,
                        states[index],
                        result["horizon"],
                        on_frame=callback,
                    )
                finally:
                    if writer:
                        writer.close()
                record = {
                    "task_id": task_id,
                    "task": task.name,
                    "episode": episode,
                    "initial_state_index": index,
                    "success": success,
                    "steps": steps,
                }
                progress.record(task.name, episode, success, record)
                print(json.dumps(record), flush=True)
        finally:
            env.close()
    summary = progress.finish()
    print(f"Mean success rate: {summary['total_success_rate']:.3f}")
    return summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained VLA policy on LIBERO or MetaWorld MT50.",
        epilog="Relative checkpoint, output and LIBERO source paths use the project root.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=["auto", "raw", "ema"], default="auto")
    parser.add_argument(
        "--output-dir", required=True,
        help="Experiment directory; reports are saved under ckpt<step>/seed<seed>",
    )
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
        "--execute-steps", type=int,
        help="Actions per observation (default: LIBERO 12, MetaWorld 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--horizon", type=int, help="Control-step limit (default: suite-specific; MetaWorld 500)"
    )
    parser.add_argument("--video", "--videos", dest="videos", action="store_true")
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
        "weights": args.weights,
        "output": args.output_dir,
        "tasks": args.tasks,
        "episodes": args.episodes if args.episodes is not None else (50 if meta else 10),
        "device_name": args.device,
        "execute_steps": (
            args.execute_steps if args.execute_steps is not None else (1 if meta else 12)
        ),
        "seed": args.seed,
        "horizon": args.horizon if args.horizon is not None else (500 if meta else None),
        "videos": args.videos,
    }
    if meta:
        from .metaworld_evaluate import evaluate_metaworld

        return evaluate_metaworld(**common, protocol=args.protocol or "lerobot")
    return evaluate(**common, suite=args.suite or "libero_spatial", libero_root=args.libero_root)


if __name__ == "__main__":
    main()
