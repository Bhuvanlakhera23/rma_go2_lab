# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate and rank RMA teacher checkpoints for Stage B qualification."""

import argparse
import contextlib
import copy
import csv
import io
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate RMA teacher checkpoints.")
parser.add_argument("--checkpoints", type=str, nargs="*", default=None, help="Explicit list of checkpoint files.")
parser.add_argument("--run-dir", type=str, default=None, help="Run directory containing model_*.pt checkpoints.")
parser.add_argument("--checkpoint-regex", type=str, default=r"model_(\d+)\.pt", help="Regex used when scanning --run-dir.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=500, help="Number of steps to evaluate per latent mode.")
parser.add_argument("--task", type=str, default="RMA-Go2-Teacher-Rough-NA", help="Task name.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV path for detailed results.")
parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth checkpoint when scanning --run-dir.")
parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of checkpoints after stride is applied.")
parser.add_argument("--quiet-policy", action="store_true", help="Suppress policy debug prints during rollout inference.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import re
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import rma_go2_lab  # noqa: F401
import isaaclab_tasks  # noqa: F401


def get_val(d, key):
    if d is None:
        return 0.0
    val = d.get(key, 0.0)
    if isinstance(val, torch.Tensor):
        return val.mean().item()
    try:
        return float(val)
    except Exception:
        return 0.0


def collect_checkpoints(explicit, run_dir, pattern, stride=1, limit=0):
    if explicit:
        return [str(Path(p)) for p in explicit]
    if not run_dir:
        raise ValueError("Provide either --checkpoints or --run-dir")
    regex = re.compile(pattern)
    found = []
    for path in Path(run_dir).glob("model_*.pt"):
        m = regex.fullmatch(path.name)
        if m:
            step = int(m.group(1))
            found.append((step, str(path)))
    found.sort()
    paths = [p for _, p in found]
    stride = max(1, int(stride))
    paths = paths[::stride]
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


def run_eval(env, runner, mode, num_steps, quiet_policy=False):
    runner.alg.policy.latent_mode = mode
    obs, _ = env.reset()

    step_reward_list = []
    episode_reward_list = []
    vel_err_list = []
    yaw_err_list = []
    base_contact_list = []
    time_out_list = []
    terrain_level_list = []
    smoothness_list = []

    with torch.no_grad():
        sink = io.StringIO()
        for _ in range(num_steps):
            if quiet_policy:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    actions = runner.alg.policy.act_inference(obs)
            else:
                actions = runner.alg.policy.act_inference(obs)
            obs, rewards, dones, infos = env.step(actions)
            step_reward_list.append(torch.mean(rewards).item())

            metrics = infos.get("metrics", {}) if isinstance(infos, dict) else {}
            logs = infos.get("log", {}) if isinstance(infos, dict) else {}

            if logs:
                episode_reward_list.append(get_val(logs, "Train/mean_reward") or get_val(logs, "Episode_Reward/track_lin_vel_xy_exp"))
                vel_err_list.append(
                    get_val(metrics, "base_velocity/error_vel_xy")
                    or get_val(logs, "Metrics/base_velocity/error_vel_xy")
                )
                yaw_err_list.append(
                    get_val(metrics, "base_velocity/error_vel_yaw")
                    or get_val(logs, "Metrics/base_velocity/error_vel_yaw")
                )
                smoothness_list.append(get_val(logs, "Episode_Reward/dof_acc_l2"))
                base_contact_list.append(get_val(logs, "Episode_Termination/base_contact"))
                time_out_list.append(get_val(logs, "Episode_Termination/time_out"))
                terrain_level_list.append(get_val(logs, "Curriculum/terrain_levels"))

    reward_source = episode_reward_list if episode_reward_list else step_reward_list

    return {
        "reward": float(np.mean(reward_source)) if reward_source else 0.0,
        "vel_err": float(np.mean(vel_err_list)) if vel_err_list else 1e3,
        "yaw_err": float(np.mean(yaw_err_list)) if yaw_err_list else 1e3,
        "smoothness": float(np.mean(smoothness_list)) if smoothness_list else 0.0,
        "base_contact": float(np.mean(base_contact_list)) if base_contact_list else 1.0,
        "time_out": float(np.mean(time_out_list)) if time_out_list else 0.0,
        "terrain_level": float(np.mean(terrain_level_list)) if terrain_level_list else 0.0,
    }


def stage_b_score(normal, zero, shuffled):
    latent_gain = (zero["vel_err"] - normal["vel_err"]) + (shuffled["vel_err"] - normal["vel_err"])
    latent_gain += 0.5 * ((normal["reward"] - zero["reward"]) + (normal["reward"] - shuffled["reward"]))
    return (
        3.0 * normal["reward"]
        + 8.0 * normal["time_out"]
        + 1.5 * normal["terrain_level"]
        + 2.0 * latent_gain
        - 10.0 * normal["base_contact"]
        - 6.0 * normal["vel_err"]
        - 1.5 * normal["yaw_err"]
    )


def flatten_result(path, normal, zero, shuffled, score):
    step_match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
    step = int(step_match.group(1)) if step_match else -1
    return {
        "checkpoint": path,
        "step": step,
        "score": score,
        "normal_reward": normal["reward"],
        "normal_vel_err": normal["vel_err"],
        "normal_yaw_err": normal["yaw_err"],
        "normal_base_contact": normal["base_contact"],
        "normal_time_out": normal["time_out"],
        "normal_terrain_level": normal["terrain_level"],
        "zero_reward": zero["reward"],
        "zero_vel_err": zero["vel_err"],
        "shuffled_reward": shuffled["reward"],
        "shuffled_vel_err": shuffled["vel_err"],
    }


def main():
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    checkpoints = collect_checkpoints(args_cli.checkpoints, args_cli.run_dir, args_cli.checkpoint_regex, stride=args_cli.stride, limit=args_cli.limit)
    if not checkpoints:
        raise RuntimeError("No checkpoints found to evaluate")

    print(f"[INFO] Evaluating {len(checkpoints)} checkpoint(s) for task {args_cli.task}")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    agent_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = agent_cfg_obj.to_dict()
    # Evaluation loads a full teacher checkpoint; do not bootstrap from the flat checkpoint first.
    agent_cfg["policy"]["pretrained_path"] = None

    results = []

    for checkpoint_path in checkpoints:
        print("\n" + "=" * 88)
        print(f"EVALUATING: {checkpoint_path}")
        print("=" * 88)
        current_agent_cfg = copy.deepcopy(agent_cfg)
        runner = OnPolicyRunner(env, current_agent_cfg, log_dir=os.path.dirname(checkpoint_path), device=args_cli.device)
        runner.load(checkpoint_path)
        runner.alg.policy.eval()

        stats_normal = run_eval(env, runner, mode="normal", num_steps=args_cli.steps, quiet_policy=args_cli.quiet_policy)
        stats_zero = run_eval(env, runner, mode="zero", num_steps=args_cli.steps, quiet_policy=args_cli.quiet_policy)
        stats_shuffled = run_eval(env, runner, mode="shuffled", num_steps=args_cli.steps, quiet_policy=args_cli.quiet_policy)
        score = stage_b_score(stats_normal, stats_zero, stats_shuffled)
        row = flatten_result(checkpoint_path, stats_normal, stats_zero, stats_shuffled, score)
        results.append(row)

        print(f"score={score:.4f} | reward={stats_normal['reward']:.4f} | vel_err={stats_normal['vel_err']:.4f} | yaw_err={stats_normal['yaw_err']:.4f} | base_contact={stats_normal['base_contact']:.4f} | time_out={stats_normal['time_out']:.4f} | terrain={stats_normal['terrain_level']:.4f}")
        print(f"latent sanity: zero vel_err={stats_zero['vel_err']:.4f}, shuffled vel_err={stats_shuffled['vel_err']:.4f}")

    results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "#" * 88)
    print("STAGE B RANKING")
    print("#" * 88)
    for idx, row in enumerate(results, start=1):
        print(
            f"{idx:>2}. step={row['step']:<5} score={row['score']:<9.4f} "
            f"reward={row['normal_reward']:<8.4f} vel_err={row['normal_vel_err']:<8.4f} "
            f"base_contact={row['normal_base_contact']:<8.4f} time_out={row['normal_time_out']:<8.4f} "
            f"terrain={row['normal_terrain_level']:<8.4f} checkpoint={row['checkpoint']}"
        )

    if args_cli.csv:
        csv_path = Path(args_cli.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[INFO] Wrote CSV results to: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
