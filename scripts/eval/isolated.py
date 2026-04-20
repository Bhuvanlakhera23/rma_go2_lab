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
parser.add_argument("--push-interval-min-s", type=float, default=None)
parser.add_argument("--push-interval-max-s", type=float, default=None)
parser.add_argument("--push-x-range", type=str, default=None, help="Comma-separated min,max for push velocity along x.")
parser.add_argument("--push-y-range", type=str, default=None, help="Comma-separated min,max for push velocity along y.")
parser.add_argument("--push-z-range", type=str, default=None, help="Comma-separated min,max for push velocity along z.")
parser.add_argument("--push-roll-range", type=str, default=None, help="Comma-separated min,max for push angular velocity around roll.")
parser.add_argument("--push-pitch-range", type=str, default=None, help="Comma-separated min,max for push angular velocity around pitch.")
parser.add_argument("--push-yaw-range", type=str, default=None, help="Comma-separated min,max for push angular velocity around yaw.")
parser.add_argument("--switch-step", type=int, default=None, help="Optional step index at which to apply a one-shot mid-episode switch.")
parser.add_argument("--switch-static-friction", type=float, default=None)
parser.add_argument("--switch-dynamic-friction", type=float, default=None)
parser.add_argument("--switch-mass-offset", type=float, default=None)
parser.add_argument("--switch-motor-stiffness-scale", type=float, default=None)
parser.add_argument("--switch-motor-damping-scale", type=float, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.managers import EventTermCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
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


def _parse_range(spec: str | None) -> tuple[float, float] | None:
    if spec is None:
        return None
    lower, upper = [part.strip() for part in spec.split(",", maxsplit=1)]
    return (float(lower), float(upper))


def _apply_push_overrides(env_cfg) -> None:
    push_ranges = {
        "x": _parse_range(args_cli.push_x_range),
        "y": _parse_range(args_cli.push_y_range),
        "z": _parse_range(args_cli.push_z_range),
        "roll": _parse_range(args_cli.push_roll_range),
        "pitch": _parse_range(args_cli.push_pitch_range),
        "yaw": _parse_range(args_cli.push_yaw_range),
    }
    push_ranges = {axis: value for axis, value in push_ranges.items() if value is not None}

    interval_min = args_cli.push_interval_min_s
    interval_max = args_cli.push_interval_max_s
    if not push_ranges and interval_min is None and interval_max is None:
        return
    if not push_ranges:
        raise ValueError("Push interval override requires at least one push axis range.")
    if interval_min is None or interval_max is None:
        raise ValueError("Push overrides require both --push-interval-min-s and --push-interval-max-s.")

    env_cfg.events.push_robot = EventTermCfg(
        func=velocity_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(interval_min, interval_max),
        params={"velocity_range": push_ranges},
    )


def _has_switch_overrides() -> bool:
    return any(
        value is not None
        for value in (
            args_cli.switch_static_friction,
            args_cli.switch_dynamic_friction,
            args_cli.switch_mass_offset,
            args_cli.switch_motor_stiffness_scale,
            args_cli.switch_motor_damping_scale,
        )
    )


def _trigger_event_term(env, term_name: str, param_overrides: dict) -> None:
    term_cfg = env.unwrapped.event_manager.get_term_cfg(term_name)
    params = dict(term_cfg.params)
    params.update(param_overrides)
    term_cfg.func(env.unwrapped, None, **params)


def _resample_material_buckets(term_cfg, static_range: tuple[float, float], dynamic_range: tuple[float, float]) -> None:
    term = term_cfg.func
    if not hasattr(term, "material_buckets"):
        return
    restitution_range = term_cfg.params.get("restitution_range", (0.0, 0.0))
    num_buckets = int(term_cfg.params.get("num_buckets", 1))
    ranges = torch.tensor(
        [static_range, dynamic_range, restitution_range],
        device="cpu",
        dtype=torch.float32,
    )
    term.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
    if term_cfg.params.get("make_consistent", False):
        term.material_buckets[:, 1] = torch.min(term.material_buckets[:, 0], term.material_buckets[:, 1])


def _apply_switch_overrides(env) -> None:
    if args_cli.switch_static_friction is not None or args_cli.switch_dynamic_friction is not None:
        param_overrides = {}
        static_range = term_dynamic_range = None
        if args_cli.switch_static_friction is not None:
            static_range = (
                args_cli.switch_static_friction,
                args_cli.switch_static_friction,
            )
            param_overrides["static_friction_range"] = static_range
        if args_cli.switch_dynamic_friction is not None:
            term_dynamic_range = (
                args_cli.switch_dynamic_friction,
                args_cli.switch_dynamic_friction,
            )
            param_overrides["dynamic_friction_range"] = term_dynamic_range
        term_cfg = env.unwrapped.event_manager.get_term_cfg("physics_material")
        if static_range is None:
            static_range = term_cfg.params["static_friction_range"]
        if term_dynamic_range is None:
            term_dynamic_range = term_cfg.params["dynamic_friction_range"]
        _resample_material_buckets(term_cfg, static_range, term_dynamic_range)
        _trigger_event_term(env, "physics_material", param_overrides)

    if args_cli.switch_mass_offset is not None:
        _trigger_event_term(
            env,
            "add_base_mass",
            {"mass_distribution_params": (args_cli.switch_mass_offset, args_cli.switch_mass_offset)},
        )

    if args_cli.switch_motor_stiffness_scale is not None or args_cli.switch_motor_damping_scale is not None:
        param_overrides = {}
        if args_cli.switch_motor_stiffness_scale is not None:
            param_overrides["stiffness_distribution_params"] = (
                args_cli.switch_motor_stiffness_scale,
                args_cli.switch_motor_stiffness_scale,
            )
        if args_cli.switch_motor_damping_scale is not None:
            param_overrides["damping_distribution_params"] = (
                args_cli.switch_motor_damping_scale,
                args_cli.switch_motor_damping_scale,
            )
        _trigger_event_term(env, "motor_strength", param_overrides)


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
    robot = env.unwrapped.scene["robot"]

    try:
        material_cfg = event_manager.get_term_cfg("physics_material")
        materials = robot.root_physx_view.get_material_properties()
        if material_cfg.params["asset_cfg"].body_ids != slice(None):
            body_ids = material_cfg.params["asset_cfg"].body_ids
            material_values = materials[:, body_ids]
        else:
            material_values = materials
        _add_stats("observed_static_friction", material_values[..., 0], checks)
        _add_stats("observed_dynamic_friction", material_values[..., 1], checks)
    except Exception:
        pass

    try:
        mass_cfg = event_manager.get_term_cfg("add_base_mass")
        masses = robot.root_physx_view.get_masses().float()
        body_ids = mass_cfg.params["asset_cfg"].body_ids
        if body_ids == slice(None):
            body_indices = torch.arange(robot.num_bodies, device=masses.device)
        else:
            body_indices = torch.as_tensor(body_ids, device=masses.device, dtype=torch.long)
        mass_offset = masses[:, body_indices] - robot.data.default_mass[:, body_indices].float()
        _add_stats("observed_mass_offset", mass_offset, checks)
    except Exception:
        pass

    try:
        actuator = robot.actuators["base_legs"]
        _add_stats("observed_motor_stiffness_scale", actuator.stiffness / 25.0, checks)
    except Exception:
        pass

    try:
        push_cfg = event_manager.get_term_cfg("push_robot")
        push_interval = getattr(push_cfg, "interval_range_s", None)
        if push_interval is not None:
            checks["observed_push_interval_min_s"] = float(push_interval[0])
            checks["observed_push_interval_max_s"] = float(push_interval[1])
    except Exception:
        pass

    return checks


def _empty_phase_stats() -> dict[str, object]:
    return {
        "reward": [],
        "vel_err": [],
        "yaw_err": [],
        "terrain_level": [],
        "valid_counts": {
            "reward": 0,
            "vel_err": 0,
            "yaw_err": 0,
            "terrain_level": 0,
        },
        "total_dones": 0,
        "total_timeouts": 0,
        "total_base_contacts": 0,
    }


def _phase_metrics(phase_stats: dict[str, object], forced_terrain_level: int | None) -> dict[str, float | None | str]:
    valid_counts = phase_stats["valid_counts"]
    terrain_level_mean = _mean(phase_stats["terrain_level"])
    terrain_level_source = "logged"
    if valid_counts["terrain_level"] == 0 and forced_terrain_level is not None and forced_terrain_level >= 0:
        terrain_level_mean = float(forced_terrain_level)
        terrain_level_source = "forced"

    total_dones = phase_stats["total_dones"]
    total_timeouts = phase_stats["total_timeouts"]
    total_base_contacts = phase_stats["total_base_contacts"]
    timeout_fraction = (total_timeouts / total_dones) if total_dones > 0 else None
    base_contact_fraction = (total_base_contacts / total_dones) if total_dones > 0 else None

    return {
        "reward_step_mean": _mean(phase_stats["reward"]),
        "vel_err_step_mean": _mean(phase_stats["vel_err"]),
        "yaw_err_step_mean": _mean(phase_stats["yaw_err"]),
        "terrain_level_step_mean": terrain_level_mean,
        "terrain_level_source": terrain_level_source,
        "terminal_dones": total_dones,
        "terminal_timeouts": total_timeouts,
        "terminal_base_contacts": total_base_contacts,
        "timeout_fraction_of_terminals": timeout_fraction,
        "base_contact_fraction_of_terminals": base_contact_fraction,
        "timeout_events_per_env": total_timeouts / max(args_cli.num_envs, 1),
        "base_contact_events_per_env": total_base_contacts / max(args_cli.num_envs, 1),
    }


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)
    _apply_randomization_overrides(env_cfg)
    _apply_push_overrides(env_cfg)

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
    switch_step = args_cli.switch_step if args_cli.switch_step is not None and args_cli.switch_step >= 0 else None
    switch_applied = False
    post_switch_constraint_checks: dict[str, float] | None = None
    overall_stats = _empty_phase_stats()
    pre_switch_stats = _empty_phase_stats()
    post_switch_stats = _empty_phase_stats()

    with torch.no_grad():
        for step_idx in range(args_cli.steps):
            if not switch_applied and switch_step is not None and step_idx == switch_step and _has_switch_overrides():
                _apply_switch_overrides(env)
                switch_applied = True
                post_switch_constraint_checks = _collect_constraint_checks(env)

            actions = runner.alg.policy.act_inference(obs)
            obs, rewards, dones, infos = env.step(actions)

            phase_stats = post_switch_stats if switch_applied else pre_switch_stats
            for stats in (overall_stats, phase_stats):
                stats["reward"].append(_to_float(rewards))
                stats["valid_counts"]["reward"] += 1

            timeouts = infos.get("time_outs", None) if isinstance(infos, dict) else None
            done_count = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else int(np.sum(dones))
            timeout_count = int(timeouts.sum().item()) if isinstance(timeouts, torch.Tensor) else 0
            base_contact_count = max(done_count - timeout_count, 0)
            for stats in (overall_stats, phase_stats):
                stats["total_dones"] += done_count
                stats["total_timeouts"] += timeout_count
                stats["total_base_contacts"] += base_contact_count

            logs = infos.get("log", {}) if isinstance(infos, dict) else {}
            metrics = infos.get("metrics", {}) if isinstance(infos, dict) else {}

            vel_err = _extract_optional_float(metrics, "base_velocity/error_vel_xy")
            if vel_err is None:
                vel_err = _extract_optional_float(logs, "Metrics/base_velocity/error_vel_xy")
            if vel_err is not None:
                for stats in (overall_stats, phase_stats):
                    stats["vel_err"].append(vel_err)
                    stats["valid_counts"]["vel_err"] += 1

            yaw_err = _extract_optional_float(metrics, "base_velocity/error_vel_yaw")
            if yaw_err is None:
                yaw_err = _extract_optional_float(logs, "Metrics/base_velocity/error_vel_yaw")
            if yaw_err is not None:
                for stats in (overall_stats, phase_stats):
                    stats["yaw_err"].append(yaw_err)
                    stats["valid_counts"]["yaw_err"] += 1

            terrain_level = _extract_optional_float(logs, "Curriculum/terrain_levels")
            if terrain_level is not None:
                for stats in (overall_stats, phase_stats):
                    stats["terrain_level"].append(terrain_level)
                    stats["valid_counts"]["terrain_level"] += 1

    overall_metrics = _phase_metrics(overall_stats, args_cli.terrain_level)
    pre_switch_metrics = _phase_metrics(pre_switch_stats, args_cli.terrain_level)
    post_switch_metrics = _phase_metrics(post_switch_stats, args_cli.terrain_level)

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
            "switch_step": args_cli.switch_step,
            "switch_static_friction": args_cli.switch_static_friction,
            "switch_dynamic_friction": args_cli.switch_dynamic_friction,
            "switch_mass_offset": args_cli.switch_mass_offset,
            "switch_motor_stiffness_scale": args_cli.switch_motor_stiffness_scale,
            "switch_motor_damping_scale": args_cli.switch_motor_damping_scale,
            "push_interval_min_s": args_cli.push_interval_min_s,
            "push_interval_max_s": args_cli.push_interval_max_s,
            "push_x_min": None if _parse_range(args_cli.push_x_range) is None else _parse_range(args_cli.push_x_range)[0],
            "push_x_max": None if _parse_range(args_cli.push_x_range) is None else _parse_range(args_cli.push_x_range)[1],
            "push_y_min": None if _parse_range(args_cli.push_y_range) is None else _parse_range(args_cli.push_y_range)[0],
            "push_y_max": None if _parse_range(args_cli.push_y_range) is None else _parse_range(args_cli.push_y_range)[1],
            "push_z_min": None if _parse_range(args_cli.push_z_range) is None else _parse_range(args_cli.push_z_range)[0],
            "push_z_max": None if _parse_range(args_cli.push_z_range) is None else _parse_range(args_cli.push_z_range)[1],
            "push_roll_min": None if _parse_range(args_cli.push_roll_range) is None else _parse_range(args_cli.push_roll_range)[0],
            "push_roll_max": None if _parse_range(args_cli.push_roll_range) is None else _parse_range(args_cli.push_roll_range)[1],
            "push_pitch_min": None if _parse_range(args_cli.push_pitch_range) is None else _parse_range(args_cli.push_pitch_range)[0],
            "push_pitch_max": None if _parse_range(args_cli.push_pitch_range) is None else _parse_range(args_cli.push_pitch_range)[1],
            "push_yaw_min": None if _parse_range(args_cli.push_yaw_range) is None else _parse_range(args_cli.push_yaw_range)[0],
            "push_yaw_max": None if _parse_range(args_cli.push_yaw_range) is None else _parse_range(args_cli.push_yaw_range)[1],
        },
        "metrics": overall_metrics,
        "pre_switch_metrics": pre_switch_metrics,
        "post_switch_metrics": post_switch_metrics,
        "constraint_checks": constraint_checks,
        "post_switch_constraint_checks": post_switch_constraint_checks,
        "valid_counts": overall_stats["valid_counts"],
        "pre_switch_valid_counts": pre_switch_stats["valid_counts"],
        "post_switch_valid_counts": post_switch_stats["valid_counts"],
        "switch_applied": switch_applied,
        "switch_applied_step": switch_step if switch_applied else None,
    }

    m = result["post_switch_metrics"] if switch_applied and post_switch_stats["valid_counts"]["reward"] > 0 else result["metrics"]
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
