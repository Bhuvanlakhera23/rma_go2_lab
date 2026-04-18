"""Headless gait diagnostics for locomotion checkpoints.

The goal is to turn visual gait checks into quantitative diagnostics. The
metrics are context-aware enough to distinguish flat-trot quality from generic
rough-terrain survival: trot scores are computed, but the final interpretation
also records command speed, terrain type, body bounce, slip, and terminations.
"""

from __future__ import annotations

import argparse
import json
import os
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

parser = argparse.ArgumentParser(description="Evaluate gait quality from contact and body-state signals.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Flat", help="Registered task name.")
parser.add_argument("--terrain-type", type=str, default=None, help="Isolated terrain name. Leave unset for task default.")
parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain level. Use -1 for task default/spread.")
parser.add_argument("--latent-mode", type=str, default="normal", choices=LATENT_MODES)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=999)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N rollout steps. Set <= 0 to disable.")
parser.add_argument("--json-out", type=str, default=None)
parser.add_argument(
    "--command-profile",
    type=str,
    default="task",
    choices=["task", "standstill", "step", "forward"],
    help="Override command generation with a controller-focused evaluation profile.",
)
parser.add_argument("--forced-lin-x", type=float, default=0.75)
parser.add_argument("--forced-lin-y", type=float, default=0.0)
parser.add_argument("--forced-ang-z", type=float, default=0.0)
parser.add_argument(
    "--step-command-start",
    type=int,
    default=200,
    help="Step index where the command switches from zero to the forced command when --command-profile=step.",
)
parser.add_argument(
    "--latency-velocity-fraction",
    type=float,
    default=0.5,
    help="Fraction of commanded planar speed used to define step-response latency.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_LABELS = ("FL", "FR", "RL", "RR")
PAIR_A = (0, 3)  # FL + RR
PAIR_B = (1, 2)  # FR + RL


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


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _ratio(a: torch.Tensor, b: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return a.float() / (b.float() + eps)


def _binary_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a == b).float()


def _binary_disagreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a != b).float()


def _extract_feet(env):
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    robot = env.unwrapped.scene["robot"]
    robot_feet = SceneEntityCfg("robot", body_names=".*_foot")
    sensor_feet = SceneEntityCfg("contact_forces", body_names=".*_foot")
    robot_feet.resolve(env.unwrapped.scene)
    sensor_feet.resolve(env.unwrapped.scene)
    foot_ids = robot_feet.body_ids
    sensor_foot_ids = sensor_feet.body_ids
    return robot, contact_sensor, foot_ids, sensor_foot_ids


def _resolve_joint_ids(env, pattern: str):
    joint_cfg = SceneEntityCfg("robot", joint_names=pattern)
    joint_cfg.resolve(env.unwrapped.scene)
    return joint_cfg.joint_ids


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, denom: torch.Tensor) -> float:
    return float((values * mask.float()).sum().item() / denom.item())


def _pair_fraction(pair_active: torch.Tensor, mask: torch.Tensor, denom: torch.Tensor) -> float:
    return float((pair_active.float() * mask.float()).sum().item() / denom.item())


def _collect_gait_metrics(env, moving_mask: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    robot, contact_sensor, foot_ids, sensor_foot_ids = _extract_feet(env)
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids, :].norm(dim=-1).max(dim=1).values
    contacts = forces > 1.0
    swing = ~contacts

    diag_a_sync = 0.5 * (_binary_agreement(contacts[:, PAIR_A[0]], contacts[:, PAIR_A[1]]))
    diag_b_sync = 0.5 * (_binary_agreement(contacts[:, PAIR_B[0]], contacts[:, PAIR_B[1]]))
    diag_sync = diag_a_sync + diag_b_sync
    diag_antiphase = _binary_disagreement(contacts[:, PAIR_A[0]], contacts[:, PAIR_B[0]])
    trot_score = diag_sync * diag_antiphase

    lateral_sync = 0.5 * (
        _binary_agreement(contacts[:, 0], contacts[:, 2]) + _binary_agreement(contacts[:, 1], contacts[:, 3])
    )
    fore_hind_sync = 0.5 * (
        _binary_agreement(contacts[:, 0], contacts[:, 1]) + _binary_agreement(contacts[:, 2], contacts[:, 3])
    )
    all_airborne = contacts.sum(dim=1) == 0
    all_stance = contacts.sum(dim=1) == 4
    swing_count = swing.sum(dim=1)
    diagonal_swing_pair = (swing[:, 0] & swing[:, 3]) | (swing[:, 1] & swing[:, 2])
    lateral_swing_pair = (swing[:, 0] & swing[:, 2]) | (swing[:, 1] & swing[:, 3])
    fore_hind_swing_pair = (swing[:, 0] & swing[:, 1]) | (swing[:, 2] & swing[:, 3])
    diagonal_swing_exact_pair = diagonal_swing_pair & (swing_count == 2)
    lateral_swing_exact_pair = lateral_swing_pair & (swing_count == 2)
    fore_hind_swing_exact_pair = fore_hind_swing_pair & (swing_count == 2)

    foot_vel_xy = robot.data.body_lin_vel_w[:, foot_ids, :2].norm(dim=-1)
    foot_slip = (foot_vel_xy * contacts.float()).sum(dim=1) / contacts.float().sum(dim=1).clamp_min(1.0)
    foot_clearance = robot.data.body_pos_w[:, foot_ids, 2] - robot.data.root_pos_w[:, 2:3]

    root_lin_z = robot.data.root_lin_vel_b[:, 2].abs()
    root_ang_xy = robot.data.root_ang_vel_b[:, :2].norm(dim=1)
    yaw_speed = robot.data.root_ang_vel_b[:, 2].abs()
    base_height = robot.data.root_pos_w[:, 2]
    command = env.unwrapped.command_manager.get_command("base_velocity")
    command_xy = command[:, :2].norm(dim=1)
    command_yaw = command[:, 2].abs()

    joint_pos = robot.data.joint_pos
    default_joint_pos = getattr(robot.data, "default_joint_pos", torch.zeros_like(joint_pos))
    joint_pos_error = (joint_pos - default_joint_pos).abs()
    joint_vel_abs = robot.data.joint_vel.abs()
    hip_ids = _resolve_joint_ids(env, ".*_hip_joint")
    thigh_ids = _resolve_joint_ids(env, ".*_thigh_joint")
    calf_ids = _resolve_joint_ids(env, ".*_calf_joint")

    projected_gravity = getattr(robot.data, "projected_gravity_b", None)
    base_tilt_xy = projected_gravity[:, :2].norm(dim=1) if projected_gravity is not None else torch.zeros_like(root_lin_z)

    mask = moving_mask
    if mask.any():
        denom = mask.float().sum()
    else:
        mask = torch.ones_like(moving_mask, dtype=torch.bool)
        denom = mask.float().sum()

    out: dict[str, float] = {
        "moving_env_fraction": float(moving_mask.float().mean().item()),
        "command_xy_mean": float(command_xy.mean().item()),
        "command_yaw_abs_mean": float(command_yaw.mean().item()),
        "diagonal_trot_score": float((trot_score * mask.float()).sum().item() / denom.item()),
        "diagonal_pair_sync_score": float((diag_sync * mask.float()).sum().item() / denom.item()),
        "diagonal_antiphase_score": float((diag_antiphase * mask.float()).sum().item() / denom.item()),
        "lateral_pair_sync_score": float((lateral_sync * mask.float()).sum().item() / denom.item()),
        "fore_hind_pair_sync_score": float((fore_hind_sync * mask.float()).sum().item() / denom.item()),
        "all_feet_airborne_fraction": float((all_airborne.float() * mask.float()).sum().item() / denom.item()),
        "all_feet_stance_fraction": float((all_stance.float() * mask.float()).sum().item() / denom.item()),
        "single_foot_swing_fraction": _pair_fraction(swing_count == 1, mask, denom),
        "two_foot_swing_fraction": _pair_fraction(swing_count == 2, mask, denom),
        "diagonal_swing_pair_fraction": _pair_fraction(diagonal_swing_pair, mask, denom),
        "diagonal_swing_exact_pair_fraction": _pair_fraction(diagonal_swing_exact_pair, mask, denom),
        "lateral_swing_exact_pair_fraction": _pair_fraction(lateral_swing_exact_pair, mask, denom),
        "fore_hind_swing_exact_pair_fraction": _pair_fraction(fore_hind_swing_exact_pair, mask, denom),
        "foot_slip_contact_mean": _masked_mean(foot_slip, mask, denom),
        "foot_clearance_abs_mean": _masked_mean(foot_clearance.abs().mean(dim=1), mask, denom),
        "base_height_mean": _masked_mean(base_height, mask, denom),
        "base_z_vel_abs_mean": _masked_mean(root_lin_z, mask, denom),
        "base_roll_pitch_ang_vel_mean": _masked_mean(root_ang_xy, mask, denom),
        "base_yaw_speed_abs_mean": _masked_mean(yaw_speed, mask, denom),
        "base_tilt_projected_gravity_xy_mean": _masked_mean(base_tilt_xy, mask, denom),
        "joint_pos_error_abs_mean": _masked_mean(joint_pos_error.mean(dim=1), mask, denom),
        "joint_pos_error_abs_max_mean": _masked_mean(joint_pos_error.max(dim=1).values, mask, denom),
        "joint_vel_abs_mean": _masked_mean(joint_vel_abs.mean(dim=1), mask, denom),
        "hip_joint_pos_error_abs_mean": _masked_mean(joint_pos_error[:, hip_ids].mean(dim=1), mask, denom),
        "thigh_joint_pos_error_abs_mean": _masked_mean(joint_pos_error[:, thigh_ids].mean(dim=1), mask, denom),
        "calf_joint_pos_error_abs_mean": _masked_mean(joint_pos_error[:, calf_ids].mean(dim=1), mask, denom),
    }

    duty = contacts.float().mean(dim=0)
    for label, value in zip(FOOT_LABELS, duty, strict=True):
        out[f"duty_factor_{label}"] = float(value.item())
    return out, contacts


def _new_event_sums() -> dict[str, float]:
    keys = [
        "touchdown_steps",
        "touchdown_single_foot_steps",
        "touchdown_diagonal_pair_steps",
        "touchdown_lateral_pair_steps",
        "touchdown_fore_hind_pair_steps",
        "liftoff_steps",
        "liftoff_single_foot_steps",
        "liftoff_diagonal_pair_steps",
        "liftoff_lateral_pair_steps",
        "liftoff_fore_hind_pair_steps",
    ]
    sums = {key: 0.0 for key in keys}
    for label in FOOT_LABELS:
        sums[f"touchdown_events_{label}"] = 0.0
        sums[f"liftoff_events_{label}"] = 0.0
    return sums


def _accumulate_contact_events(
    event_sums: dict[str, float], prev_contacts: torch.Tensor, contacts: torch.Tensor, moving_mask: torch.Tensor
) -> None:
    mask = moving_mask.bool()
    if not mask.any():
        return

    touchdown = (~prev_contacts) & contacts
    liftoff = prev_contacts & (~contacts)

    def _accumulate(prefix: str, events: torch.Tensor) -> None:
        count = events.sum(dim=1)
        event_sums[f"{prefix}_steps"] += float(((count > 0) & mask).sum().item())
        event_sums[f"{prefix}_single_foot_steps"] += float(((count == 1) & mask).sum().item())
        event_sums[f"{prefix}_diagonal_pair_steps"] += float(
            ((((events[:, 0] & events[:, 3]) | (events[:, 1] & events[:, 2])) & (count == 2) & mask).sum().item())
        )
        event_sums[f"{prefix}_lateral_pair_steps"] += float(
            ((((events[:, 0] & events[:, 2]) | (events[:, 1] & events[:, 3])) & (count == 2) & mask).sum().item())
        )
        event_sums[f"{prefix}_fore_hind_pair_steps"] += float(
            ((((events[:, 0] & events[:, 1]) | (events[:, 2] & events[:, 3])) & (count == 2) & mask).sum().item())
        )
        for index, label in enumerate(FOOT_LABELS):
            event_sums[f"{prefix}_events_{label}"] += float((events[:, index].float() * mask.float()).sum().item())

    _accumulate("touchdown", touchdown)
    _accumulate("liftoff", liftoff)


def _finalize_event_metrics(event_sums: dict[str, float], num_envs: int, steps: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix in ("touchdown", "liftoff"):
        total_steps = event_sums[f"{prefix}_steps"]
        metrics[f"{prefix}_event_steps_per_env"] = total_steps / max(num_envs, 1)
        metrics[f"{prefix}_event_step_fraction"] = total_steps / max(num_envs * steps, 1)
        for kind in ("single_foot", "diagonal_pair", "lateral_pair", "fore_hind_pair"):
            metrics[f"{prefix}_{kind}_step_fraction_of_events"] = (
                event_sums[f"{prefix}_{kind}_steps"] / total_steps if total_steps > 0 else 0.0
            )
        total_foot_events = sum(event_sums[f"{prefix}_events_{label}"] for label in FOOT_LABELS)
        for label in FOOT_LABELS:
            metrics[f"{prefix}_event_share_{label}"] = (
                event_sums[f"{prefix}_events_{label}"] / total_foot_events if total_foot_events > 0 else 0.0
            )
    return metrics


def _gait_interpretation(metrics: dict[str, float], terrain_type: str | None) -> str:
    if terrain_type in (None, "plane", "random_rough"):
        if (
            metrics["diagonal_swing_exact_pair_fraction"] > 0.25
            and metrics["touchdown_diagonal_pair_step_fraction_of_events"] > 0.25
            and metrics["all_feet_airborne_fraction"] < 0.08
            and metrics["base_z_vel_abs_mean"] < 0.15
        ):
            return "event_trot_like"
        if (
            metrics["diagonal_swing_exact_pair_fraction"] > 0.25
            and metrics["lateral_swing_exact_pair_fraction"] < 0.05
            and metrics["fore_hind_swing_exact_pair_fraction"] < 0.05
            and metrics["touchdown_single_foot_step_fraction_of_events"] > 0.5
            and metrics["all_feet_airborne_fraction"] < 0.08
            and metrics["base_z_vel_abs_mean"] < 0.15
        ):
            return "high_duty_diagonal_gait_staggered_touchdown"
        if (
            metrics["diagonal_trot_score"] > 0.65
            and metrics["all_feet_airborne_fraction"] < 0.08
            and metrics["base_z_vel_abs_mean"] < 0.15
        ):
            return "trot_like"
        if metrics["all_feet_airborne_fraction"] > 0.15 or metrics["base_z_vel_abs_mean"] > 0.25:
            return "hop_or_bounce_like"
        if metrics["diagonal_trot_score"] < 0.4:
            return "non_trot_or_serial"
    return "context_dependent"


def _set_forced_velocity_command(env, command: torch.Tensor) -> None:
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:, :] = command
    command_term.time_left[:] = 1.0e6
    command_term.command_counter[:] = 1
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "heading_target"):
        command_term.heading_target[:] = env.unwrapped.scene["robot"].data.heading_w


def _command_for_step(env, step_idx: int) -> torch.Tensor | None:
    if args_cli.command_profile == "task":
        return None

    robot = env.unwrapped.scene["robot"]
    command = torch.zeros((env.num_envs, 3), device=robot.device, dtype=robot.data.root_lin_vel_b.dtype)
    if args_cli.command_profile == "standstill":
        return command
    if args_cli.command_profile == "forward":
        command[:, 0] = args_cli.forced_lin_x
        command[:, 1] = args_cli.forced_lin_y
        command[:, 2] = args_cli.forced_ang_z
        return command
    if args_cli.command_profile == "step":
        if step_idx >= args_cli.step_command_start:
            command[:, 0] = args_cli.forced_lin_x
            command[:, 1] = args_cli.forced_lin_y
            command[:, 2] = args_cli.forced_ang_z
        return command
    raise ValueError(f"Unsupported command profile: {args_cli.command_profile}")


def _collect_controller_metrics(
    env,
    actions: torch.Tensor,
    prev_actions: torch.Tensor | None,
    step_idx: int,
    response_crossings: torch.Tensor | None,
) -> tuple[dict[str, float], torch.Tensor | None]:
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")
    planar_speed_cmd = command[:, :2].norm(dim=1)
    planar_speed = robot.data.root_lin_vel_b[:, :2].norm(dim=1)
    foot_cfg = SceneEntityCfg("robot", body_names=".*_foot")
    foot_cfg.resolve(env.unwrapped.scene)
    foot_speed = robot.data.body_lin_vel_w[:, foot_cfg.body_ids, :].norm(dim=-1).mean(dim=1)
    joint_vel_abs = robot.data.joint_vel.abs().mean(dim=1)
    body_y = robot.data.root_pos_w[:, 1]
    heading = robot.data.heading_w

    metrics: dict[str, float] = {
        "planar_speed_mean": float(planar_speed.mean().item()),
        "command_planar_speed_mean": float(planar_speed_cmd.mean().item()),
        "standstill_planar_speed_mean": float(planar_speed[planar_speed_cmd < 0.1].mean().item())
        if (planar_speed_cmd < 0.1).any()
        else 0.0,
        "standstill_foot_speed_mean": float(foot_speed[planar_speed_cmd < 0.1].mean().item())
        if (planar_speed_cmd < 0.1).any()
        else 0.0,
        "standstill_joint_vel_abs_mean": float(joint_vel_abs[planar_speed_cmd < 0.1].mean().item())
        if (planar_speed_cmd < 0.1).any()
        else 0.0,
        "heading_abs_mean": float(heading.abs().mean().item()),
        "lateral_position_abs_mean": float(body_y.abs().mean().item()),
    }

    if prev_actions is not None:
        metrics["action_delta_abs_mean"] = float((actions - prev_actions).abs().mean().item())
    else:
        metrics["action_delta_abs_mean"] = 0.0

    if args_cli.command_profile == "forward":
        x_disp = robot.data.root_pos_w[:, 0] - env.unwrapped.scene.env_origins[:, 0]
        y_disp = robot.data.root_pos_w[:, 1] - env.unwrapped.scene.env_origins[:, 1]
        metrics["forward_heading_abs_mean"] = float(heading.abs().mean().item())
        metrics["forward_lateral_drift_abs_mean"] = float(y_disp.abs().mean().item())
        metrics["forward_lateral_drift_per_meter_mean"] = float((y_disp.abs() / x_disp.abs().clamp_min(0.1)).mean().item())

    if args_cli.command_profile == "step":
        after_step = step_idx >= args_cli.step_command_start
        target_speed = max((args_cli.forced_lin_x**2 + args_cli.forced_lin_y**2) ** 0.5, 1.0e-6)
        threshold = args_cli.latency_velocity_fraction * target_speed
        if after_step and response_crossings is not None:
            crossed = (response_crossings < 0) & (planar_speed >= threshold)
            response_crossings[crossed] = step_idx
            valid = response_crossings >= 0
            metrics["step_response_reached_fraction"] = float(valid.float().mean().item())
            metrics["step_response_latency_steps_mean"] = float(
                (response_crossings[valid] - args_cli.step_command_start).float().mean().item()
            ) if valid.any() else float(args_cli.steps - args_cli.step_command_start)
            metrics["step_response_planar_speed_mean"] = float(planar_speed.mean().item())
            metrics["step_response_speed_ratio_mean"] = float((planar_speed / target_speed).mean().item())
            metrics["step_response_overshoot_mean"] = float(
                torch.clamp((planar_speed - target_speed) / target_speed, min=0.0).mean().item()
            )
        else:
            metrics["step_response_reached_fraction"] = 0.0
            metrics["step_response_latency_steps_mean"] = 0.0
            metrics["step_response_planar_speed_mean"] = float(planar_speed.mean().item())
            metrics["step_response_speed_ratio_mean"] = float((planar_speed / target_speed).mean().item())
            metrics["step_response_overshoot_mean"] = 0.0

    return metrics, response_crossings


def main() -> None:
    env = None
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)

    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
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

        env.reset()
        _force_terrain_level(env, args_cli.terrain_level)
        forced_command = _command_for_step(env, step_idx=0)
        if forced_command is not None:
            _set_forced_velocity_command(env, forced_command)
        obs = env.get_observations()

        metric_sums: dict[str, float] = {}
        event_sums = _new_event_sums()
        prev_contacts: torch.Tensor | None = None
        prev_actions: torch.Tensor | None = None
        response_crossings: torch.Tensor | None = None
        if args_cli.command_profile == "step":
            response_crossings = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        total_dones = 0
        total_timeouts = 0
        total_base_contacts = 0

        with torch.inference_mode():
            for step_idx in range(args_cli.steps):
                if args_cli.progress_every > 0 and (step_idx == 0 or (step_idx + 1) % args_cli.progress_every == 0):
                    print(f"[INFO] gait.py progress: step {step_idx + 1}/{args_cli.steps}", flush=True)
                forced_command = _command_for_step(env, step_idx=step_idx)
                if forced_command is not None:
                    _set_forced_velocity_command(env, forced_command)
                    obs = env.get_observations()
                actions = runner.alg.policy.act_inference(obs)
                obs, _, dones, infos = env.step(actions)

                command = env.unwrapped.command_manager.get_command("base_velocity")
                moving_mask = command[:, :2].norm(dim=1) > 0.15
                step_metrics, contacts = _collect_gait_metrics(env, moving_mask)
                controller_metrics, response_crossings = _collect_controller_metrics(
                    env, actions, prev_actions, step_idx, response_crossings
                )
                if prev_contacts is not None:
                    _accumulate_contact_events(event_sums, prev_contacts, contacts, moving_mask)
                prev_contacts = contacts.clone()
                prev_actions = actions.clone()
                for key, value in step_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value
                for key, value in controller_metrics.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value

                timeouts = infos.get("time_outs", None) if isinstance(infos, dict) else None
                done_count = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else 0
                timeout_count = int(timeouts.sum().item()) if isinstance(timeouts, torch.Tensor) else 0
                total_dones += done_count
                total_timeouts += timeout_count
                total_base_contacts += max(done_count - timeout_count, 0)

        metrics = {key: value / max(args_cli.steps, 1) for key, value in metric_sums.items()}
        metrics["terminal_dones"] = total_dones
        metrics["terminal_timeouts"] = total_timeouts
        metrics["terminal_base_contacts"] = total_base_contacts
        metrics["timeout_fraction_of_terminals"] = total_timeouts / total_dones if total_dones > 0 else None
        metrics["base_contact_fraction_of_terminals"] = total_base_contacts / total_dones if total_dones > 0 else None
        metrics["base_contact_events_per_env"] = total_base_contacts / max(args_cli.num_envs, 1)
        metrics["timeout_events_per_env"] = total_timeouts / max(args_cli.num_envs, 1)
        metrics.update(_finalize_event_metrics(event_sums, args_cli.num_envs, args_cli.steps))
        metrics["gait_interpretation"] = _gait_interpretation(metrics, args_cli.terrain_type)

        result = {
            "checkpoint": args_cli.checkpoint,
            "task": args_cli.task,
            "terrain_type": args_cli.terrain_type,
            "terrain_level": args_cli.terrain_level,
            "latent_mode": args_cli.latent_mode,
            "command_profile": args_cli.command_profile,
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "metrics": metrics,
        }

        print(json.dumps(result, indent=2))

        json_out = args_cli.json_out
        if json_out is None:
            ckpt = Path(args_cli.checkpoint).stem
            terrain = args_cli.terrain_type or "default"
            level = f"level{args_cli.terrain_level}" if args_cli.terrain_level >= 0 else "levelspread"
            json_out = str(
                REPO_ROOT
                / "artifacts/evaluations"
                / f"gait_{ckpt}_{args_cli.task}_{terrain}_{level}_{args_cli.latent_mode}_seed{args_cli.seed}.json"
            )
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote JSON to: {json_out}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
