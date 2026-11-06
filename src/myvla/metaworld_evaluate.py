"""MetaWorld v3 RGB/state rollouts, isolated from the LIBERO runtime."""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .metaworld_data import task_definitions, task_language
from .preprocess import decode_actions, normalize_state, prepare_image
from .train import policy_from_checkpoint, resolve_device, seed_everything


def import_metaworld():
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import metaworld
    except ImportError as exc:
        raise RuntimeError("Use the isolated MT50 environment: uv sync --extra mt50") from exc
    return metaworld


def make_env(name, protocol="lerobot", benchmark=None):
    metaworld = import_metaworld()
    mt = benchmark if benchmark is not None else metaworld.MT1(name, seed=42)
    goals = [task for task in mt.train_tasks if task.env_name == name]
    env = mt.train_classes[name](
        render_mode="rgb_array", camera_name="corner2", width=480, height=480
    )
    try:
        env.set_task(goals[0])
        # Match the official LeRobot MetaWorld wrapper's camera, with both-axis flip.
        env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        if protocol == "lerobot":
            env.reset()
            env._freeze_rand_vec = False
            env.seeded_rand_vec = True
        elif protocol != "benchmark":
            raise ValueError("protocol must be lerobot or benchmark")
    except BaseException:
        env.close()
        raise
    return env, goals


def render_rgb(env):
    return np.ascontiguousarray(np.flip(env.render(), (0, 1)))


class MetaWorldController:
    def __init__(self, policy, config, stats, device, execute_steps):
        if not 1 <= execute_steps <= config.model.chunk_size:
            raise ValueError("execute_steps must be in [1, chunk_size]")
        self.policy, self.config, self.stats = policy, config, stats
        self.device, self.execute_steps = device, execute_steps
        self.queue = []

    def reset(self):
        self.queue.clear()

    @torch.no_grad()
    def act(self, obs, env, language):
        if not self.queue:
            # Only gripper xyz and opening; exclude object/goal privileged state.
            state = normalize_state(np.asarray(obs, np.float32)[:4], self.stats)
            image = prepare_image(render_rgb(env), self.config.model.image_size, False)
            pred = self.policy(
                image[None, None].to(self.device),
                torch.from_numpy(state)[None].to(self.device),
                [language],
            )
            actions = decode_actions(pred[0].float().cpu().numpy(), self.stats)
            self.queue = list(actions[: self.execute_steps])
        return self.queue.pop(0)


def run_episode(env, controller, language, seed, horizon=500, on_frame=None):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    controller.reset()
    # MetaWorld 3.0 ignores reset(seed=...), so seed its own RNG explicitly.
    env.seed(seed)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    for step in range(horizon):
        if on_frame:
            on_frame(render_rgb(env))
        obs, reward, terminated, truncated, info = env.step(controller.act(obs, env, language))
        total_reward += float(reward)
        # Success at any step counts, irrespective of terminal reward.
        if bool(info.get("success", False)):
            return True, step + 1, total_reward
        if terminated or truncated:
            break
    return False, step + 1, total_reward


def evaluate_metaworld(
    checkpoint,
    output,
    tasks=None,
    episodes=50,
    device_name="cuda",
    execute_steps=1,
    seed=42,
    horizon=500,
    protocol="lerobot",
    videos=False,
    weights="auto",
):
    if episodes < 1 or not 1 <= horizon <= 500:
        raise ValueError("episodes must be positive; horizon must be in [1,500]")
    if protocol not in {"lerobot", "benchmark"}:
        raise ValueError("protocol must be lerobot or benchmark")
    if protocol == "benchmark" and episodes > 50:
        raise ValueError("Benchmark evaluation has only 50 distinct goal positions per task")
    definitions = task_definitions()
    by_id = {i: name for name, i in definitions["task_ids"].items()}
    task_ids = sorted(by_id) if tasks is None else tasks
    if not task_ids or len(set(task_ids)) != len(task_ids) or not set(task_ids) <= by_id.keys():
        raise ValueError("Select unique task IDs in [0,49]")
    out = Path(output)
    if any((out / name).exists() for name in ("episodes.jsonl", "results.json")):
        raise FileExistsError(f"Evaluation output exists: {out}")
    device = resolve_device(device_name)
    seed_everything(seed)
    policy, config, stats, payload = policy_from_checkpoint(checkpoint, device, weights=weights)
    if config.data.dataset != "metaworld" or config.model.backend != "clip":
        raise ValueError("MetaWorld benchmark requires a real CLIP MT50 checkpoint")
    controller = MetaWorldController(policy, config, stats, device, execute_steps)
    trained_languages = {ep["language"] for ep in payload["manifest"] if ep["split"] == "train"}
    trained_tasks = [name for name in by_id.values() if task_language(name) in trained_languages]
    metaworld = import_metaworld()
    benchmark = metaworld.MT50(seed=seed) if protocol == "benchmark" else None
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "suite": "metaworld_mt50",
        "protocol": protocol,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_step": payload["step"],
        "weights": payload["inference_weights"],
        "config": config.to_dict(),
        "seed": seed,
        "episodes_per_task": episodes,
        "horizon": horizon,
        "execute_steps": execute_steps,
        "camera": {
            "name": "corner2",
            "position": [0.75, 0.075, 0.7],
            "width": 480,
            "height": 480,
            "flip_both_axes": True,
        },
        "state": "raw_observation[:4]",
        "control": "continuous xyz + gripper [-1,1]",
        "success_criterion": "info.success at any step",
        "tasks": [],
        "full_mt50": len(task_ids) == 50,
        "training_task_count": len(trained_tasks),
        "training_tasks": trained_tasks,
        "versions": {
            k: version(k)
            for k in ("myvla", "metaworld", "mujoco", "gymnasium", "numpy", "torch", "transformers")
        },
    }
    with (out / "episodes.jsonl").open("w", buffering=1) as log:
        for task_id in task_ids:
            name = by_id[task_id]
            env, goals = make_env(name, protocol, benchmark)
            successes = 0
            rng = np.random.default_rng(np.random.SeedSequence([seed, task_id]))
            goal_indices = rng.permutation(len(goals))[:episodes].tolist()
            try:
                for episode in range(episodes):
                    episode_seed = seed + 10000 * task_id + episode
                    goal_index, goal_hash = None, None
                    if protocol == "benchmark":
                        goal_index = goal_indices[episode]
                        goal = goals[goal_index]
                        env.set_task(goal)
                        goal_hash = hashlib.sha256(goal.data).hexdigest()
                    writer = None
                    try:
                        if videos:
                            import imageio.v2 as imageio

                            writer = imageio.get_writer(
                                str(out / f"task{task_id:02d}_episode{episode:03d}.mp4"), fps=80
                            )
                        success, steps, reward = run_episode(
                            env,
                            controller,
                            task_language(name),
                            episode_seed,
                            horizon,
                            on_frame=writer.append_data if writer else None,
                        )
                    finally:
                        if writer:
                            writer.close()
                    successes += int(success)
                    record = {
                        "task_id": task_id,
                        "task": name,
                        "episode": episode,
                        "seed": episode_seed,
                        "goal_index": goal_index,
                        "goal_sha256": goal_hash,
                        "success": success,
                        "steps": steps,
                        "return": reward,
                    }
                    log.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
            finally:
                env.close()
            result["tasks"].append(
                {
                    "task_id": task_id,
                    "task": name,
                    "episodes": episodes,
                    "successes": successes,
                    "success_rate": successes / episodes,
                }
            )
            # Keep completed-task summaries if a later task is interrupted.
            result["completed"] = False
            temp = out / "results.json.tmp"
            temp.write_text(json.dumps(result, indent=2))
            temp.replace(out / "results.json")
    result["completed"] = True
    result["mean_success_rate"] = float(np.mean([t["success_rate"] for t in result["tasks"]]))
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Mean success rate: {result['mean_success_rate']:.3f}")
    return result
