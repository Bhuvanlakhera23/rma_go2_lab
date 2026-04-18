"""Evaluate a checkpoint on isolated terrain with controlled randomization overrides."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate checkpoint under isolated and deterministic conditions.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Blind-Baseline-Rough-WarmStart", help="Registered task name.")
parser.add_argument("--terrain-type", type=str, default="random_rough", help="Isolated terrain name.")
parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain curriculum level. Use -1 for spread up to max_init_terrain_level.")
LATENT_MODES = [
    "normal",
    "zero",
    "shuffled",
    "no_terrain",
    "shuffled_terrain",
    "no_dynamics",
    "shuffled_dynamics",
    "no_history",
    "random",
    "shuffled_time",
]
parser.add_argument("--latent-mode", type=str, default="normal", choices=LATENT_MODES)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output file path. Defaults to artifacts/evaluations/<auto>.json.")

# Optional deterministic overrides (min=max)
parser.add_argument("--static-friction", type=float, default=None)
parser.add_argument("--dynamic-friction", type=float, default=None)
parser.add_argument("--mass-offset", type=float, default=None)
parser.add_argument("--motor-stiffness-scale", type=float, default=None)
parser.add_argument("--motor-damping-scale", type=float, default=None)

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
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]


def _force_isolated_terrain(env_cfg, terrain_type: str) -> None:
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


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _to_float(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.mean().item())
    try:
        return float(v)
    except Exception:
        return 0.0


def _extract_optional_float(container: dict, key: str) -> float | None:
    if not isinstance(container, dict) or key not in container:
        return None
    value = container.get(key)
    try:
        return _to_float(value)
    except Exception:
        return None


def _tensor_stats(value) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.float().reshape(-1)
        if tensor.numel() == 0:
            return None
        return {
            "mean": float(tensor.mean().item()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
        }
    except Exception:
        return None


def _add_stats(prefix: str, value, out: dict[str, float]) -> None:
    stats = _tensor_stats(value)
    if stats is None:
        return
    out[f"{prefix}_mean"] = stats["mean"]
    out[f"{prefix}_min"] = stats["min"]
    out[f"{prefix}_max"] = stats["max"]


def _collect_constraint_checks(env) -> dict[str, float]:
    """Read back applied randomization values to catch no-op override bugs."""
    checks: dict[str, float] = {}
    event_manager = env.unwrapped.event_manager

    if "physics_material" in event_manager.active_terms:
        term = event_manager.get_term("physics_material")
        _add_stats("observed_static_friction", getattr(term, "static_friction", None), checks)
        _add_stats("observed_dynamic_friction", getattr(term, "dynamic_friction", None), checks)

    if "add_base_mass" in event_manager.active_terms:
        term = event_manager.get_term("add_base_mass")
        _add_stats("observed_mass_offset", getattr(term, "mass_offset", None), checks)

    try:
        actuator = env.unwrapped.scene["robot"].actuators["base_legs"]
        _add_stats("observed_motor_stiffness_scale", actuator.stiffness / 25.0, checks)
    except Exception:
        pass

    return checks


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
    runner.alg.policy.eval()
    if hasattr(runner.alg.policy, "latent_mode"):
        runner.alg.policy.latent_mode = args_cli.latent_mode

    obs, _ = env.reset()
    _force_terrain_level(env, args_cli.terrain_level)
    obs = env.get_observations()
    constraint_checks = _collect_constraint_checks(env)
    reward_list: list[float] = []
    vel_err_list: list[float] = []
    yaw_err_list: list[float] = []
    terrain_level_list: list[float] = []
    total_dones = 0
    total_timeouts = 0
    total_base_contacts = 0
    valid_counts = {
        "reward": 0,
        "vel_err": 0,
        "yaw_err": 0,
        "terrain_level": 0,
    }

    with torch.no_grad():
        for _ in range(args_cli.steps):
            actions = runner.alg.policy.act_inference(obs)
            obs, rewards, dones, infos = env.step(actions)
            reward_list.append(_to_float(rewards))
            valid_counts["reward"] += 1

            timeouts = infos.get("time_outs", None) if isinstance(infos, dict) else None
            done_count = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else int(np.sum(dones))
            timeout_count = int(timeouts.sum().item()) if isinstance(timeouts, torch.Tensor) else 0
            base_contact_count = max(done_count - timeout_count, 0)
            total_dones += done_count
            total_timeouts += timeout_count
            total_base_contacts += base_contact_count

            logs = infos.get("log", {}) if isinstance(infos, dict) else {}
            metrics = infos.get("metrics", {}) if isinstance(infos, dict) else {}

            vel_err = _extract_optional_float(metrics, "base_velocity/error_vel_xy")
            if vel_err is None:
                vel_err = _extract_optional_float(logs, "Metrics/base_velocity/error_vel_xy")
            if vel_err is not None:
                vel_err_list.append(vel_err)
                valid_counts["vel_err"] += 1

            yaw_err = _extract_optional_float(metrics, "base_velocity/error_vel_yaw")
            if yaw_err is None:
                yaw_err = _extract_optional_float(logs, "Metrics/base_velocity/error_vel_yaw")
            if yaw_err is not None:
                yaw_err_list.append(yaw_err)
                valid_counts["yaw_err"] += 1

            terrain_level = _extract_optional_float(logs, "Curriculum/terrain_levels")
            if terrain_level is not None:
                terrain_level_list.append(terrain_level)
                valid_counts["terrain_level"] += 1

    terrain_level_mean = _mean(terrain_level_list)
    terrain_level_source = "logged"
    if valid_counts["terrain_level"] == 0 and args_cli.terrain_level is not None and args_cli.terrain_level >= 0:
        terrain_level_mean = float(args_cli.terrain_level)
        terrain_level_source = "forced"

    timeout_fraction = (total_timeouts / total_dones) if total_dones > 0 else None
    base_contact_fraction = (total_base_contacts / total_dones) if total_dones > 0 else None
    base_contact_events_per_env = total_base_contacts / max(args_cli.num_envs, 1)
    timeout_events_per_env = total_timeouts / max(args_cli.num_envs, 1)

    result = {
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "terrain_type": args_cli.terrain_type,
        "terrain_level": args_cli.terrain_level,
        "latent_mode": args_cli.latent_mode,
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "overrides": {
            "static_friction": args_cli.static_friction,
            "dynamic_friction": args_cli.dynamic_friction,
            "mass_offset": args_cli.mass_offset,
            "motor_stiffness_scale": args_cli.motor_stiffness_scale,
            "motor_damping_scale": args_cli.motor_damping_scale,
        },
        "metrics": {
            "reward_step_mean": _mean(reward_list),
            "vel_err_step_mean": _mean(vel_err_list),
            "yaw_err_step_mean": _mean(yaw_err_list),
            "terrain_level_step_mean": terrain_level_mean,
            "terrain_level_source": terrain_level_source,
            "terminal_dones": total_dones,
            "terminal_timeouts": total_timeouts,
            "terminal_base_contacts": total_base_contacts,
            "timeout_fraction_of_terminals": timeout_fraction,
            "base_contact_fraction_of_terminals": base_contact_fraction,
            "timeout_events_per_env": timeout_events_per_env,
            "base_contact_events_per_env": base_contact_events_per_env,
        },
        "constraint_checks": constraint_checks,
        "valid_counts": valid_counts,
    }

    m = result["metrics"]
    score = (
        3.0 * m["reward_step_mean"]
        + 1.5 * m["terrain_level_step_mean"]
        + 6.0 * (m["timeout_fraction_of_terminals"] if m["timeout_fraction_of_terminals"] is not None else 0.0)
        - 10.0 * m["base_contact_events_per_env"]
        - 6.0 * m["vel_err_step_mean"]
        - 1.5 * m["yaw_err_step_mean"]
    )
    result["score"] = float(score)

    print("\n=== Isolated Evaluation Result ===")
    print(json.dumps(result, indent=2))

    json_out = args_cli.json_out
    if json_out is None:
        ckpt = Path(args_cli.checkpoint).stem
        suffix = args_cli.terrain_type
        if args_cli.static_friction is not None:
            suffix += f"_fric{args_cli.static_friction:g}"
        json_out = str(REPO_ROOT / "artifacts/evaluations" / f"isolated_{ckpt}_{suffix}_{args_cli.latent_mode}_seed{args_cli.seed}.json")

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Wrote JSON to: {json_out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
