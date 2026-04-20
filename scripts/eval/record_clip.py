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
parser.add_argument("--push-interval-min-s", type=float, default=None, help="Min interval between pushes in seconds.")
parser.add_argument("--push-interval-max-s", type=float, default=None, help="Max interval between pushes in seconds.")
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
parser.add_argument(
    "--disable-terminations",
    type=str,
    default="",
    help="Comma-separated termination term names to disable for visualization-only recording, e.g. 'base_height,low_progress'.",
)
parser.add_argument(
    "--command-profile",
    type=str,
    default="task",
    choices=["task", "standstill", "forward"],
    help="Optionally override the sampled command with a fixed controller-focused profile.",
)
parser.add_argument("--forced-lin-x", type=float, default=0.75, help="Forced x velocity when using --command-profile forward.")
parser.add_argument("--forced-lin-y", type=float, default=0.0, help="Forced y velocity when using --command-profile forward.")
parser.add_argument("--forced-ang-z", type=float, default=0.0, help="Forced yaw velocity when using --command-profile forward.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401
from rma_go2_lab.models.blind.frozen_flat_expert import FrozenFlatExpert


def _safe_load_runner(runner: OnPolicyRunner, checkpoint_path: str) -> None:
    try:
        runner.load(checkpoint_path)
        return
    except RuntimeError as exc:
        message = str(exc)
        if "normalizer" not in message:
            raise
        print("[WARN] Standard runner.load() failed due to normalizer-key mismatch.")
        print("[WARN] Retrying with checkpoint normalizer entries filtered out for playback.")

    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    model_state = checkpoint["model_state_dict"]
    filtered_state = {k: v for k, v in model_state.items() if "normalizer" not in k}
    resumed = runner.alg.policy.load_state_dict(filtered_state, strict=False)
    print(f"[INFO] Fallback policy load complete. Missing keys: {list(resumed.missing_keys)}")
    print(f"[INFO] Fallback policy load complete. Unexpected keys: {list(resumed.unexpected_keys)}")


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
        static_range = dynamic_range = None
        if args_cli.switch_static_friction is not None:
            static_range = (
                args_cli.switch_static_friction,
                args_cli.switch_static_friction,
            )
            param_overrides["static_friction_range"] = static_range
        if args_cli.switch_dynamic_friction is not None:
            dynamic_range = (
                args_cli.switch_dynamic_friction,
                args_cli.switch_dynamic_friction,
            )
            param_overrides["dynamic_friction_range"] = dynamic_range
        term_cfg = env.unwrapped.event_manager.get_term_cfg("physics_material")
        if static_range is None:
            static_range = term_cfg.params["static_friction_range"]
        if dynamic_range is None:
            dynamic_range = term_cfg.params["dynamic_friction_range"]
        _resample_material_buckets(term_cfg, static_range, dynamic_range)
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


def _disabled_termination_names() -> list[str]:
    if not args_cli.disable_terminations.strip():
        return []
    return [name.strip() for name in args_cli.disable_terminations.split(",") if name.strip()]


def _apply_termination_overrides(env_cfg) -> None:
    disabled = _disabled_termination_names()
    if not disabled:
        return
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    available = {
        name
        for name in dir(terminations)
        if not name.startswith("_") and getattr(terminations, name) is not None
    }
    unknown = [name for name in disabled if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown termination term(s) {unknown}. Available active terms include: {sorted(available)}"
        )
    for name in disabled:
        setattr(terminations, name, None)
    print(f"[INFO] Disabled termination terms for recording only: {disabled}")


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


def _forced_command(env) -> torch.Tensor | None:
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
    raise ValueError(f"Unsupported command profile: {args_cli.command_profile}")


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
        "disabled_terminations": _disabled_termination_names(),
        "command_profile": args_cli.command_profile,
        "forced_command": {
            "lin_x": args_cli.forced_lin_x,
            "lin_y": args_cli.forced_lin_y,
            "ang_z": args_cli.forced_ang_z,
        },
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
    _apply_push_overrides(env_cfg)
    _apply_termination_overrides(env_cfg)

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
    video_env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)
    _force_terrain_level(env, args_cli.terrain_level)

    use_direct_flat_expert = args_cli.task == "RMA-Go2-Flat"
    policy_nn = None
    if use_direct_flat_expert:
        print("[INFO] Using direct FrozenFlatExpert playback path for flat prior recording.")
        flat_expert = FrozenFlatExpert(
            checkpoint_path=args_cli.checkpoint,
            activation="elu",
            device=str(env.unwrapped.device),
        ).to(env.unwrapped.device)

        def policy(obs):
            return flat_expert(obs["policy"])

    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        _safe_load_runner(runner, args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        policy_nn = runner.alg.policy
        if hasattr(policy_nn, "latent_mode"):
            policy_nn.latent_mode = args_cli.latent_mode

    obs = env.get_observations()
    forced_command = _forced_command(env)
    if forced_command is not None:
        _set_forced_velocity_command(env, forced_command)
        obs = env.get_observations()
    # Ensure the recorder sees an initial frame before stepping.
    try:
        video_env.render()
    except Exception as exc:  # pragma: no cover - best-effort recorder guard
        print(f"[WARN] Initial render failed before recording: {exc}")
    dt = env.unwrapped.step_dt
    expected_seconds = args_cli.video_length * dt
    print(
        "[INFO] Recording clip "
        f"'{args_cli.tag}' for {args_cli.video_length} steps "
        f"(~{expected_seconds:.1f}s sim time) into {run_dir}"
    )

    progress_interval = max(1, args_cli.video_length // 4)
    switch_step = args_cli.switch_step if args_cli.switch_step is not None and args_cli.switch_step >= 0 else None
    switch_applied = False
    for timestep in range(1, args_cli.video_length + 1):
        if not simulation_app.is_running():
            print("[WARN] Simulation app stopped before recording completed.")
            break
        with torch.inference_mode():
            if not switch_applied and switch_step is not None and timestep == switch_step and _has_switch_overrides():
                _apply_switch_overrides(env)
                switch_applied = True
            if forced_command is not None:
                _set_forced_velocity_command(env, forced_command)
                obs = env.get_observations()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if policy_nn is not None:
                policy_nn.reset(dones)
        try:
            video_env.render()
        except Exception as exc:  # pragma: no cover - best-effort recorder guard
            print(f"[WARN] Render failed during recording step {timestep}: {exc}")
            break

        if timestep % progress_interval == 0 or timestep == args_cli.video_length:
            print(f"[INFO] Recording progress: {timestep}/{args_cli.video_length} steps")

        if args_cli.real_time:
            import time
            time.sleep(dt)

    video_env.close()

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
