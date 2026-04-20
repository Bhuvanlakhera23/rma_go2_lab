"""Play or record the frozen flat prior with a flat-specific loader.

This avoids the generic RSL-RL runner load path, which currently conflicts
with the saved flat checkpoint contents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play the frozen flat prior with a flat-specific loader.")
parser.add_argument("--checkpoint", type=str, required=True, help="Flat-prior checkpoint file path.")
parser.add_argument("--task", type=str, default="RMA-Go2-Flat", help="Registered task name.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=1200, help="Recorded video length in env steps.")
parser.add_argument(
    "--video_folder",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "artifacts/evaluations/clips/flat_prior"),
    help="Folder used when --video is enabled.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import isaaclab_tasks  # noqa: F401
import rma_go2_lab  # noqa: F401
from rma_go2_lab.models.blind.frozen_flat_expert import FrozenFlatExpert


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_folder = Path(args_cli.video_folder).resolve()
        video_folder.mkdir(parents=True, exist_ok=True)
        video_kwargs = {
            "video_folder": str(video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "fps": int(1 / env.unwrapped.step_dt),
            "disable_logger": True,
        }
        print("[INFO] Recording flat-prior playback video.")
        print(f"[INFO] video_folder: {video_folder}")
        print(f"[INFO] video_length: {args_cli.video_length}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    flat_expert = FrozenFlatExpert(
        checkpoint_path=args_cli.checkpoint,
        activation="elu",
        device=str(env.unwrapped.device),
    ).to(env.unwrapped.device)

    print(f"[INFO] Loaded flat prior from: {args_cli.checkpoint}")

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0

    while simulation_app.is_running():
        start_time = None
        if args_cli.real_time:
            import time

            start_time = time.time()

        with torch.inference_mode():
            actions = flat_expert(obs["policy"])
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        if args_cli.real_time:
            import time

            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
