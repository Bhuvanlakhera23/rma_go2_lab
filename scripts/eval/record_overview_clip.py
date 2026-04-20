"""Record an overview playback clip while cycling the camera across env indices."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Record a multi-env overview clip with automatic env-index cycling.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--task", type=str, required=True, help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video in steps.")
parser.add_argument(
    "--env_indices",
    type=str,
    default="0,30,60,90",
    help="Comma-separated env indices to cycle through, e.g. '0,30,60,90'.",
)
parser.add_argument(
    "--switch_interval",
    type=int,
    default=200,
    help="Number of env steps to hold each env index before switching.",
)
parser.add_argument("--tag", type=str, default="overview", help="Suffix used in the archived clip name.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "artifacts/evaluations/clips"),
    help="Directory to store the uniquely named mp4.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--terrain-type", type=str, default=None, help="Isolated terrain name (e.g. random_rough, hf_pyramid_slope).")
parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain curriculum level. Use -1 for task default.")
parser.add_argument("--static-friction", type=float, default=None, help="Set static friction to a fixed value.")
parser.add_argument("--dynamic-friction", type=float, default=None, help="Set dynamic friction to a fixed value.")
parser.add_argument("--mass-offset", type=float, default=None, help="Set base mass offset to a fixed value.")
parser.add_argument("--motor-stiffness-scale", type=float, default=None, help="Scale actuator stiffness.")
parser.add_argument("--motor-damping-scale", type=float, default=None, help="Scale actuator damping.")
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

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
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


def _parse_env_indices(spec: str) -> list[int]:
    indices = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        indices.append(int(chunk))
    if not indices:
        raise ValueError("Expected at least one env index in --env_indices.")
    return indices


def _set_view_env(env, env_index: int) -> None:
    controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    if controller is None:
        return
    controller.set_view_env_index(env_index)
    controller.update_view_to_asset_root("robot")


def _write_metadata(run_dir: Path, run_stamp: str, env_indices: list[int]) -> None:
    metadata = {
        "artifact_type": "overview_clip",
        "tag": args_cli.tag,
        "timestamp": run_stamp,
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "video_length": args_cli.video_length,
        "headless": bool(getattr(args_cli, "headless", False)),
        "real_time": args_cli.real_time,
        "env_indices": env_indices,
        "switch_interval": args_cli.switch_interval,
        "terrain_type": args_cli.terrain_type,
        "terrain_level": None if args_cli.terrain_level < 0 else args_cli.terrain_level,
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
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    env_indices = _parse_env_indices(args_cli.env_indices)

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    _force_isolated_terrain(env_cfg, args_cli.terrain_type)
    _disable_terrain_curriculum_for_fixed_level(env_cfg, args_cli.terrain_level)
    _apply_randomization_overrides(env_cfg)
    _apply_termination_overrides(env_cfg)

    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{args_cli.tag}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(run_dir, run_stamp, env_indices)

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

    obs = env.get_observations()
    forced_command = _forced_command(env)
    if forced_command is not None:
        _set_forced_velocity_command(env, forced_command)
        obs = env.get_observations()
    try:
        video_env.render()
    except Exception as exc:  # pragma: no cover - best-effort recorder guard
        print(f"[WARN] Initial render failed before recording: {exc}")
    dt = env.unwrapped.step_dt
    current_index_slot = 0
    _set_view_env(env, env_indices[current_index_slot])
    expected_seconds = args_cli.video_length * dt
    print(
        "[INFO] Recording overview clip "
        f"'{args_cli.tag}' for {args_cli.video_length} steps "
        f"(~{expected_seconds:.1f}s sim time) into {run_dir}"
    )
    print(f"[INFO] Cycling env indices: {env_indices} every {args_cli.switch_interval} steps")

    progress_interval = max(1, args_cli.video_length // 4)
    for timestep in range(1, args_cli.video_length + 1):
        if not simulation_app.is_running():
            print("[WARN] Simulation app stopped before recording completed.")
            break
        with torch.inference_mode():
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

        if timestep % args_cli.switch_interval == 0:
            current_index_slot = (current_index_slot + 1) % len(env_indices)
            _set_view_env(env, env_indices[current_index_slot])
        if timestep % progress_interval == 0 or timestep == args_cli.video_length:
            print(f"[INFO] Recording progress: {timestep}/{args_cli.video_length} steps")

        if args_cli.real_time:
            import time

            time.sleep(dt)

    video_env.close()

    mp4 = run_dir / "rl-video-step-0.mp4"
    if mp4.exists():
        final_mp4 = run_dir / f"{args_cli.tag}_{run_stamp}.mp4"
        mp4.rename(final_mp4)
        print(f"[INFO] Saved overview clip to: {final_mp4}")
    else:
        print(f"[WARN] Expected recorded mp4 not found in {run_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
