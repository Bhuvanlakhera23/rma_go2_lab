"""Quantitative diagnostics for blind-baseline locomotion checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate reward and termination health for a blind locomotion baseline.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Blind-Baseline-Rough", help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--json-out", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _unwrap_obs(obs):
    """Handle IsaacLab/RSL-RL wrappers that may return tuple or bare obs."""
    if isinstance(obs, tuple):
        return obs[0]
    return obs


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    reward_names = list(env.unwrapped.reward_manager.active_terms)
    termination_names = list(env.unwrapped.termination_manager.active_terms)

    reward_signed_sums = {name: 0.0 for name in reward_names}
    reward_abs_sums = {name: 0.0 for name in reward_names}
    termination_counts = {name: 0 for name in termination_names}

    obs = _unwrap_obs(env.get_observations())
    live_episode_steps = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.long)
    completed_episode_lengths: list[int] = []

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        obs = _unwrap_obs(obs)

        live_episode_steps += 1

        step_reward = env.unwrapped.reward_manager._step_reward
        for idx, name in enumerate(reward_names):
            term_values = step_reward[:, idx]
            reward_signed_sums[name] += float(term_values.mean().item())
            reward_abs_sums[name] += float(term_values.abs().mean().item())

        for name in termination_names:
            fired = env.unwrapped.termination_manager.get_term(name)
            termination_counts[name] += int(fired.sum().item())

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            completed_episode_lengths.extend(live_episode_steps[done_ids].cpu().tolist())
            live_episode_steps[done_ids] = 0

    results = {
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "reward_mean_per_step": {
            name: reward_signed_sums[name] / args_cli.steps for name in reward_names
        },
        "reward_abs_mean_per_step": {
            name: reward_abs_sums[name] / args_cli.steps for name in reward_names
        },
        "termination_counts": termination_counts,
        "termination_fraction_of_env_steps": {
            name: termination_counts[name] / float(args_cli.num_envs * args_cli.steps)
            for name in termination_names
        },
        "episode_length_mean": _mean_or_zero([float(v) for v in completed_episode_lengths]),
        "episode_length_first_100_mean": _mean_or_zero([float(v) for v in completed_episode_lengths[:100]]),
        "episode_length_last_100_mean": _mean_or_zero([float(v) for v in completed_episode_lengths[-100:]]),
        "num_completed_episodes": len(completed_episode_lengths),
    }

    print(json.dumps(results, indent=2))
    if args_cli.json_out:
        Path(args_cli.json_out).write_text(json.dumps(results, indent=2))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
