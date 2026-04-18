"""Record a uniquely named playback clip for a checkpoint.

Unlike IsaacLab's stock `play.py --video`, this script:

- saves to a unique repo-local filename
- supports fixed terrain / DR overrides like `play_isolated.py`
- avoids overwriting `rl-video-step-0.mp4` between runs
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher


LATENT_MODES = [
    "normal",
    "zero",
    "shuffled",
    "no_terrain",
    "shuffled_terrain",
    "no_dynamics",
    "shuffled_dynamics",
]

parser = argparse.ArgumentParser(description="Record a uniquely named playback clip for a checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=999, help="Seed used for the environment.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video in steps.")
parser.add_argument("--tag", type=str, default="clip", help="Suffix used in the archived clip name.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "artifacts/evaluations/clips"),
    help="Directory where the uniquely named copy is archived.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--terrain-type", type=str, default=None, help="Isolated terrain name (e.g. random_rough, hf_pyramid_slope).")
parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain curriculum level. Use -1 for task default.")
parser.add_argument("--latent-mode", type=str, default="normal", choices=LATENT_MODES)
parser.add_argument("--static-friction", type=float, default=None, help="Set static friction to a fixed value.")
parser.add_argument("--dynamic-friction", type=float, default=None, help="Set dynamic friction to a fixed value.")
parser.add_argument("--mass-offset", type=float, default=None, help="Set base mass offset to a fixed value.")
parser.add_argument("--motor-stiffness-scale", type=float, default=None, help="Scale actuator stiffness.")
parser.add_argument("--motor-damping-scale", type=float, default=None, help="Scale actuator damping.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


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
    if args_cli.static_friction is not None and env_cfg.events.physics_material is not None:
        env_cfg.events.physics_material.params["static_friction_range"] = (
            args_cli.static_friction,
            args_cli.static_friction,
        )
    if args_cli.dynamic_friction is not None and env_cfg.events.physics_material is not None:
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


def _write_metadata(run_dir: Path, timestamp: str) -> None:
    metadata = {
        "artifact_type": "clip",
        "tag": args_cli.tag,
        "timestamp": timestamp,
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "video_length": args_cli.video_length,
        "headless": bool(getattr(args_cli, "headless", False)),
        "real_time": args_cli.real_time,
        "terrain_type": args_cli.terrain_type,
        "terrain_level": None if args_cli.terrain_level < 0 else args_cli.terrain_level,
        "latent_mode": args_cli.latent_mode,
        "overrides": {
            "static_friction": args_cli.static_friction,
            "dynamic_friction": args_cli.dynamic_friction,
            "mass_offset": args_cli.mass_offset,
            "motor_stiffness_scale": args_cli.motor_stiffness_scale,
            "motor_damping_scale": args_cli.motor_damping_scale,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)
    _apply_randomization_overrides(env_cfg)

    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{args_cli.tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(run_dir, timestamp)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    video_kwargs = {
        "video_folder": str(run_dir),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "fps": 30,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _force_terrain_level(env, args_cli.terrain_level)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy
    if hasattr(policy_nn, "latent_mode"):
        policy_nn.latent_mode = args_cli.latent_mode

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        timestep += 1
        if timestep >= args_cli.video_length:
            break

        if args_cli.real_time:
            import time
            time.sleep(dt)

    env.close()

    mp4 = run_dir / "rl-video-step-0.mp4"
    if mp4.exists():
        final_mp4 = run_dir / f"{args_cli.tag}_{timestamp}.mp4"
        mp4.rename(final_mp4)
        print(f"[INFO] Saved clip to: {final_mp4}")
    else:
        print(f"[WARN] Expected recorded mp4 not found in {run_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
