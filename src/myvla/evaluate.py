import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .libero_runtime import load_init_states, setup_libero
from .preprocess import decode_actions, normalize_state, prepare_image, state_from_observation
from .train import policy_from_checkpoint, resolve_device, seed_everything

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
    policy, config, stats, payload = policy_from_checkpoint(checkpoint, device)
    if config.model.backend != "clip":
        raise ValueError("Tiny random smoke model cannot be used for benchmark reporting")
    controller = Controller(policy, config, stats, device, execute_steps)
    benchmark_suite = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    task_ids = list(range(benchmark_suite.n_tasks)) if tasks is None else tasks
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Select at least one task, without duplicate IDs")
    if any(i < 0 or i >= benchmark_suite.n_tasks for i in task_ids):
        raise ValueError("Task ID out of bounds")
    out = Path(output)
    if (out / "episodes.jsonl").exists() or (out / "results.json").exists():
        raise FileExistsError(f"Evaluation output exists: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_step": payload["step"],
        "suite": suite,
        "seed": seed,
        "episodes_per_task": episodes,
        "execute_steps": execute_steps,
        "horizon": horizon or HORIZONS[suite],
        "settle_steps": 10,
        "control": "relative OSC_POSE",
        "config": config.to_dict(),
        "libero": source_info,
        "tasks": [],
    }
    result["versions"] = {
        name: version(name)
        for name in ("myvla", "torch", "transformers", "robosuite", "mujoco", "numpy")
    }
    with (out / "episodes.jsonl").open("w", buffering=1) as log:
        for task_id in task_ids:
            task = benchmark_suite.get_task(task_id)
            states, states_sha256 = load_init_states(benchmark_suite, task_id)
            if episodes > len(states):
                raise ValueError(
                    f"Requested {episodes} unique episodes; only {len(states)} init states"
                )
            # Same task/seed selects the same unique initial states across model runs.
            rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
            indices = rng.permutation(len(states))[:episodes].tolist()
            env = OffScreenRenderEnv(
                bddl_file_name=benchmark_suite.get_task_bddl_file_path(task_id),
                camera_heights=128,
                camera_widths=128,
            )
            successes = 0
            try:
                for episode, index in enumerate(indices):
                    env.seed(seed + task_id * 10000 + index)
                    writer = None
                    try:
                        callback = None
                        if videos:
                            import imageio.v2 as imageio

                            writer = imageio.get_writer(
                                str(out / f"task{task_id:02d}_episode{episode:03d}.mp4"), fps=20
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
                    successes += int(success)
                    record = {
                        "task_id": task_id,
                        "task": task.name,
                        "episode": episode,
                        "initial_state_index": index,
                        "success": success,
                        "steps": steps,
                    }
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
            finally:
                env.close()
            result["tasks"].append(
                {
                    "task_id": task_id,
                    "task": task.name,
                    "initial_states_sha256": states_sha256,
                    "successes": successes,
                    "episodes": episodes,
                    "success_rate": successes / episodes,
                }
            )
            result["mean_success_rate"] = float(
                np.mean([task["success_rate"] for task in result["tasks"]])
            )
            (out / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Mean success rate: {result['mean_success_rate']:.3f}")
    return result
