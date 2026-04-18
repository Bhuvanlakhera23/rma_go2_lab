"""Play an RMA checkpoint on isolated terrains with controlled randomization."""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play checkpoint with isolated terrain and controlled dynamics.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Blind-Baseline-Rough-WarmStart", help="Registered task name.")
parser.add_argument("--terrain-type", type=str, default=None, help="Isolated terrain name (e.g. pyramid_stairs, boxes).")
parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain curriculum level. Use -1 for spread up to max_init_terrain_level.")
LATENT_MODES = [
    "normal",
    "zero",
    "shuffled",
    "no_terrain",
    "shuffled_terrain",
    "no_dynamics",
    "shuffled_dynamics",
]
parser.add_argument("--latent-mode", type=str, default="normal", choices=LATENT_MODES)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--steps", type=int, default=-1, help="Max steps. Use -1 to run until window closes.")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match simulator dt.")

# Optional deterministic overrides (min=max)
parser.add_argument("--static-friction", type=float, default=None, help="Set static friction to a fixed value.")
parser.add_argument("--dynamic-friction", type=float, default=None, help="Set dynamic friction to a fixed value.")
parser.add_argument("--mass-offset", type=float, default=None, help="Set base mass offset to a fixed value.")
parser.add_argument("--motor-stiffness-scale", type=float, default=None, help="Scale actuator stiffness.")
parser.add_argument("--motor-damping-scale", type=float, default=None, help="Scale actuator damping.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import rma_go2_lab  # noqa: F401
import isaaclab_tasks  # noqa: F401


def _force_isolated_terrain(env_cfg, terrain_type: str | None) -> None:
    if terrain_type is None:
        return
    env_cfg.scene.terrain.max_init_terrain_level = 9
    terrain_gen = env_cfg.scene.terrain.terrain_generator
    for key in terrain_gen.sub_terrains.keys():
        terrain_gen.sub_terrains[key].proportion = 0.0
    if terrain_type not in terrain_gen.sub_terrains:
        valid = list(terrain_gen.sub_terrains.keys())
        raise ValueError(f"Unknown terrain type '{terrain_type}'. Valid options: {valid}")
    terrain_gen.sub_terrains[terrain_type].proportion = 1.0




def _disable_terrain_curriculum_for_fixed_level(env_cfg, terrain_level: int | None) -> None:
    if terrain_level is None or terrain_level < 0:
        return
    if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

def _force_terrain_level(env, terrain_level: int | None) -> None:
    if terrain_level is None or terrain_level < 0:
        return
    terrain = env.unwrapped.scene.terrain
    if getattr(terrain, "terrain_origins", None) is None:
        return
    level = max(0, min(int(terrain_level), int(terrain.terrain_origins.shape[0]) - 1))
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.unwrapped.scene.env_origins[:] = terrain.env_origins

def _apply_randomization_overrides(env_cfg) -> None:
    if (
        args_cli.static_friction is not None
        and env_cfg.events.physics_material is not None
    ):
        env_cfg.events.physics_material.params["static_friction_range"] = (
            args_cli.static_friction,
            args_cli.static_friction,
        )
    if (
        args_cli.dynamic_friction is not None
        and env_cfg.events.physics_material is not None
    ):
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (
            args_cli.dynamic_friction,
            args_cli.dynamic_friction,
        )
    if args_cli.mass_offset is not None and env_cfg.events.add_base_mass is not None:
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
            args_cli.mass_offset,
            args_cli.mass_offset,
        )
    if args_cli.motor_stiffness_scale is not None and hasattr(env_cfg.events, "motor_strength"):
        env_cfg.events.motor_strength.params["stiffness_distribution_params"] = (
            args_cli.motor_stiffness_scale,
            args_cli.motor_stiffness_scale,
        )
    if args_cli.motor_damping_scale is not None and hasattr(env_cfg.events, "motor_strength"):
        env_cfg.events.motor_strength.params["damping_distribution_params"] = (
            args_cli.motor_damping_scale,
            args_cli.motor_damping_scale,
        )


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)
    _apply_randomization_overrides(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    _force_terrain_level(env, args_cli.terrain_level)

    agent_cfg_obj = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = agent_cfg_obj.to_dict()
    if "policy" in agent_cfg and isinstance(agent_cfg["policy"], dict):
        agent_cfg["policy"]["pretrained_path"] = None

    runner = OnPolicyRunner(env, agent_cfg, log_dir=os.path.dirname(args_cli.checkpoint), device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    if hasattr(policy_nn, "latent_mode"):
        policy_nn.latent_mode = args_cli.latent_mode

    print("=== Play Configuration ===")
    print(f"task={args_cli.task}")
    print(f"checkpoint={args_cli.checkpoint}")
    print(f"terrain_type={args_cli.terrain_type}")
    print(f"terrain_level={args_cli.terrain_level}")
    print(f"latent_mode={args_cli.latent_mode}")
    print(f"num_envs={args_cli.num_envs}, seed={args_cli.seed}")
    print(
        "overrides: "
        f"static_friction={args_cli.static_friction}, dynamic_friction={args_cli.dynamic_friction}, "
        f"mass_offset={args_cli.mass_offset}, stiffness_scale={args_cli.motor_stiffness_scale}, "
        f"damping_scale={args_cli.motor_damping_scale}"
    )

    _force_terrain_level(env, args_cli.terrain_level)
    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    step_count = 0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        step_count += 1
        if args_cli.steps > 0 and step_count >= args_cli.steps:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
