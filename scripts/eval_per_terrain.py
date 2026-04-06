# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate RMA teacher per terrain.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--terrain-type", type=str, required=True, help="Terrain to isolate (e.g. pyramid_stairs, boxes).")
parser.add_argument("--latent-mode", type=str, default="normal", help="normal, zero, or shuffled")
parser.add_argument("--steps", type=int, default=500, help="Evaluation steps.")
parser.add_argument("--seed", type=int, default=999, help="Random seed for repeatable terrain.")
parser.add_argument("--task", type=str, default="RMA-Go2-Teacher-Rough-NA")
parser.add_argument("--num_envs", type=int, default=64)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rma_go2_lab  # noqa: F401
import isaaclab_tasks  # noqa: F401

def main():
    print(f"\n==========================================================")
    print(f"STRESS TEST: {args_cli.terrain_type.upper()} | Z-MODE: {args_cli.latent_mode.upper()}")
    print(f"==========================================================\n")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # 1. FORCE MAXIMUM DIFFICULTY TERRAIN
    env_cfg.scene.terrain.max_init_terrain_level = 9
    
    # 2. FORCE ISOLATED TERRAIN TYPE
    terrain_gen = env_cfg.scene.terrain.terrain_generator
    # Zero out all proportions
    for k in terrain_gen.sub_terrains.keys():
        terrain_gen.sub_terrains[k].proportion = 0.0
    
    # Set the target terrain to 100%
    if args_cli.terrain_type in terrain_gen.sub_terrains:
        terrain_gen.sub_terrains[args_cli.terrain_type].proportion = 1.0
    else:
        raise ValueError(f"Unknown terrain type '{args_cli.terrain_type}'. Valid: {list(terrain_gen.sub_terrains.keys())}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    agent_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = agent_cfg_obj.to_dict()
    agent_cfg["policy"]["pretrained_path"] = None

    runner = OnPolicyRunner(env, agent_cfg, log_dir=os.path.dirname(args_cli.checkpoint), device="cuda:0")
    runner.load(args_cli.checkpoint)
    runner.alg.policy.eval()
    
    # Override latent mode!
    runner.alg.policy.latent_mode = args_cli.latent_mode

    obs, _ = env.reset()
    
    base_contact_list = []
    vel_err_list = []
    
    with torch.no_grad():
        step_count = 0
        while args_cli.steps < 0 or step_count < args_cli.steps:
            actions = runner.alg.policy.act_inference(obs)
            obs, rewards, dones, infos = env.step(actions)
            step_count += 1
            
            logs = infos.get("log", {}) if isinstance(infos, dict) else {}
            metrics = infos.get("metrics", {}) if isinstance(infos, dict) else {}
            
            if logs:
                base_contact_list.append(logs.get("Episode_Termination/base_contact", 0.0))
                vel_err = metrics.get("base_velocity/error_vel_xy", logs.get("Metrics/base_velocity/error_vel_xy", 0.0))
                vel_err_list.append(vel_err)

    def get_mean(lst):
        filtered = [v.mean().item() if isinstance(v, torch.Tensor) else float(v) for v in lst if v is not None]
        return sum(filtered) / len(filtered) if filtered else 0.0

    final_contact = get_mean(base_contact_list)
    final_vel_err = get_mean(vel_err_list)

    print(f"\n---> RESULTS FOR {args_cli.terrain_type} ({args_cli.latent_mode}) <---")
    print(f"Fall Rate (base_contact): {final_contact * 100:.2f}%")
    print(f"Tracking Error (vel_err): {final_vel_err:.4f}\n")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
