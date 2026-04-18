"""Compact run/config summary for locomotion experiments.

This script consolidates the values that most strongly shape behavior in a run:

- observations and action space
- commands
- terrain and curriculum
- domain-randomization events
- rewards
- terminations
- PPO / policy settings

It is designed to work well with IsaacLab log directories (`params/env.yaml`,
`params/agent.yaml`) but can also summarize a registered task directly when run
inside the IsaacLab Python environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    if isinstance(value, slice):
        return f"slice({value.start},{value.stop},{value.step})"
    if value is None:
        return "-"
    return str(value)


def _short_func_name(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    if ":" in text:
        return text.split(":")[-1]
    return text


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered_rows = [[_format_scalar(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    lines = [
        render_row(headers),
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    lines.extend(render_row(row) for row in rendered_rows)
    return "\n".join(lines)


def _strip_cfg_noise(params: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    ignore_keys = {
        "asset_cfg",
        "sensor_cfg",
        "body_names",
        "body_ids",
        "joint_names",
        "joint_ids",
        "name",
        "preserve_order",
    }
    return {key: value for key, value in params.items() if key not in ignore_keys}


def _collect_observations(env_cfg: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    observations = env_cfg.get("observations", {})
    for group_name, group_cfg in observations.items():
        if not isinstance(group_cfg, dict):
            continue
        for term_name, term_cfg in group_cfg.items():
            if term_name in {
                "concatenate_terms",
                "concatenate_dim",
                "enable_corruption",
                "history_length",
                "flatten_history_dim",
            }:
                continue
            if term_cfg is None:
                rows.append([group_name, term_name, "disabled", "-", "-", "-"])
                continue
            noise = term_cfg.get("noise")
            noise_desc = "-"
            if isinstance(noise, dict):
                noise_desc = f"{_short_func_name(noise.get('func'))}({_format_scalar(noise.get('n_min'))},{_format_scalar(noise.get('n_max'))})"
            rows.append(
                [
                    group_name,
                    term_name,
                    "active",
                    _short_func_name(term_cfg.get("func")),
                    noise_desc,
                    _format_scalar(term_cfg.get("clip")),
                ]
            )
    return rows


def _collect_rewards(env_cfg: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    rewards = env_cfg.get("rewards", {})
    for name, cfg in rewards.items():
        if cfg is None:
            rows.append([name, "disabled", "-", "-", "-"])
            continue
        rows.append(
            [
                name,
                _format_scalar(cfg.get("weight")),
                _short_func_name(cfg.get("func")),
                json.dumps(_strip_cfg_noise(cfg.get("params")), default=str),
                "task" if float(cfg.get("weight", 0.0)) > 0 else "regularizer",
            ]
        )
    return rows


def _collect_terminations(env_cfg: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    terminations = env_cfg.get("terminations", {})
    for name, cfg in terminations.items():
        if cfg is None:
            rows.append([name, "disabled", "-", "-", "-"])
            continue
        rows.append(
            [
                name,
                "time_out" if cfg.get("time_out") else "termination",
                _short_func_name(cfg.get("func")),
                json.dumps(_strip_cfg_noise(cfg.get("params")), default=str),
                _format_scalar(cfg.get("time_out")),
            ]
        )
    return rows


def _collect_events(env_cfg: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    events = env_cfg.get("events", {})
    for name, cfg in events.items():
        if cfg is None:
            rows.append([name, "disabled", "-", "-", "-"])
            continue
        rows.append(
            [
                name,
                _format_scalar(cfg.get("mode")),
                _short_func_name(cfg.get("func")),
                json.dumps(_strip_cfg_noise(cfg.get("params")), default=str),
                _format_scalar(cfg.get("interval_range_s")),
            ]
        )
    return rows


def _collect_terrain(env_cfg: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    terrain = env_cfg.get("scene", {}).get("terrain", {})
    terrain_gen = terrain.get("terrain_generator")
    if terrain_gen is None:
        return [[terrain.get("terrain_type", "unknown"), "-", "-", "-", "-"]]

    sub_terrains = terrain_gen.get("sub_terrains", {})
    for name, cfg in sub_terrains.items():
        proportion = cfg.get("proportion", 0.0)
        if proportion in (0, 0.0, None):
            continue
        detail = "-"
        for key in ("step_height_range", "grid_height_range", "noise_range", "slope_range"):
            if key in cfg:
                detail = f"{key}={_format_scalar(cfg[key])}"
                break
        rows.append(
            [
                name,
                _format_scalar(proportion),
                detail,
                _format_scalar(cfg.get("noise_step")),
                _format_scalar(terrain.get("max_init_terrain_level")),
            ]
        )
    return rows


def _collect_commands(env_cfg: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    commands = env_cfg.get("commands", {})
    for name, cfg in commands.items():
        if cfg is None:
            continue
        ranges = cfg.get("ranges", {})
        rows.append(
            [
                name,
                _format_scalar(ranges.get("lin_vel_x")),
                _format_scalar(ranges.get("lin_vel_y")),
                _format_scalar(ranges.get("ang_vel_z")),
                _format_scalar(ranges.get("heading")),
                _format_scalar(cfg.get("heading_command")),
            ]
        )
    return rows


def _collect_ppo(agent_cfg: dict[str, Any]) -> list[list[Any]]:
    policy = agent_cfg.get("policy", {})
    algorithm = agent_cfg.get("algorithm", {})
    rows = [
        ["runner_class", agent_cfg.get("class_name")],
        ["experiment_name", agent_cfg.get("experiment_name")],
        ["seed", agent_cfg.get("seed")],
        ["device", agent_cfg.get("device")],
        ["num_steps_per_env", agent_cfg.get("num_steps_per_env")],
        ["max_iterations", agent_cfg.get("max_iterations")],
        ["save_interval", agent_cfg.get("save_interval")],
        ["policy_class", policy.get("class_name")],
        ["actor_hidden_dims", policy.get("actor_hidden_dims")],
        ["critic_hidden_dims", policy.get("critic_hidden_dims")],
        ["init_noise_std", policy.get("init_noise_std")],
        ["algorithm_class", algorithm.get("class_name")],
        ["learning_rate", algorithm.get("learning_rate")],
        ["entropy_coef", algorithm.get("entropy_coef")],
        ["clip_param", algorithm.get("clip_param")],
        ["gamma", algorithm.get("gamma")],
        ["lam", algorithm.get("lam")],
        ["desired_kl", algorithm.get("desired_kl")],
        ["num_learning_epochs", algorithm.get("num_learning_epochs")],
        ["num_mini_batches", algorithm.get("num_mini_batches")],
        ["value_loss_coef", algorithm.get("value_loss_coef")],
        ["max_grad_norm", algorithm.get("max_grad_norm")],
    ]
    for key in (
        "actor_init_path",
        "flat_expert_path",
        "flat_imitation_coef_stage0",
        "flat_imitation_coef_stage1",
        "flat_imitation_stage0_end",
        "flat_imitation_stage1_end",
    ):
        if key in policy:
            rows.append([key, policy.get(key)])
        if key in algorithm:
            rows.append([key, algorithm.get(key)])
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        # IsaacLab dumps configclasses with Python-specific tags such as
        # tuples and slices, so the stricter loaders cannot reconstruct them.
        return yaml.unsafe_load(file)


def _resolve_run_dir(run_dir: str | None, experiment: str | None) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    if experiment:
        root = Path("logs") / "rsl_rl" / experiment
        if not root.exists():
            root = Path("/home/bhuvan/tools/IsaacLab/logs/rsl_rl") / experiment
        if not root.exists():
            raise FileNotFoundError(f"Could not find experiment directory for '{experiment}'.")
        candidates = sorted(path for path in root.iterdir() if path.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No run directories found under '{root}'.")
        return candidates[-1]
    raise ValueError("Provide either --run-dir, --experiment, or --task.")


def _load_from_task(task: str, agent_entry: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    import isaaclab_tasks  # noqa: F401
    import rma_go2_lab  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, agent_entry)
    return env_cfg.to_dict(), agent_cfg.to_dict(), f"task:{task}"


def build_report(env_cfg: dict[str, Any], agent_cfg: dict[str, Any], source_label: str) -> str:
    lines: list[str] = []
    lines.append(f"# Run Summary: {source_label}")
    lines.append("")

    meta_rows = [
        ["num_envs", env_cfg.get("scene", {}).get("num_envs")],
        ["episode_length_s", env_cfg.get("episode_length_s")],
        ["sim_dt", env_cfg.get("sim", {}).get("dt")],
        ["decimation", env_cfg.get("decimation")],
        ["device", agent_cfg.get("device", env_cfg.get("sim", {}).get("device"))],
        ["seed", env_cfg.get("seed", agent_cfg.get("seed"))],
    ]
    lines.append("## Meta")
    lines.append(_markdown_table(["Field", "Value"], meta_rows))
    lines.append("")

    lines.append("## PPO")
    lines.append(_markdown_table(["Setting", "Value"], _collect_ppo(agent_cfg)))
    lines.append("")

    lines.append("## Commands")
    lines.append(_markdown_table(["Command", "lin_vel_x", "lin_vel_y", "ang_vel_z", "heading", "heading_command"], _collect_commands(env_cfg)))
    lines.append("")

    lines.append("## Observations")
    lines.append(_markdown_table(["Group", "Term", "Status", "Func", "Noise", "Clip"], _collect_observations(env_cfg)))
    lines.append("")

    curriculum = env_cfg.get("curriculum", {})
    curriculum_rows = [["terrain_levels_func", _short_func_name((curriculum.get("terrain_levels") or {}).get("func"))]]
    lines.append("## Terrain / Curriculum")
    terrain_rows = _collect_terrain(env_cfg)
    lines.append(_markdown_table(["Terrain", "Proportion", "Detail", "Noise Step", "Max Init Level"], terrain_rows))
    lines.append("")
    lines.append(_markdown_table(["Curriculum Setting", "Value"], curriculum_rows))
    lines.append("")

    lines.append("## Domain Randomization")
    lines.append(_markdown_table(["Event", "Mode", "Func", "Key Params", "Interval"], _collect_events(env_cfg)))
    lines.append("")

    lines.append("## Rewards")
    lines.append(_markdown_table(["Reward", "Weight", "Func", "Key Params", "Role"], _collect_rewards(env_cfg)))
    lines.append("")

    lines.append("## Terminations")
    lines.append(_markdown_table(["Termination", "Kind", "Func", "Key Params", "Time Out"], _collect_terminations(env_cfg)))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact summary of locomotion run settings.")

    # --- Your args ONLY ---
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Run directory containing params/env.yaml and params/agent.yaml.")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name under logs/rsl_rl; latest run is used.")
    parser.add_argument("--task", type=str, default=None,
                        help="Registered task name. Use with IsaacLab Python.")
    parser.add_argument("--agent-entry", type=str, default="rsl_rl_cfg_entry_point",
                        help="Agent registry entry when using --task.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output markdown path.")

    # --- IsaacLab args (DO NOT redefine them manually) ---
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    simulation_app = None

    # -------------------------------
    # TASK MODE (requires Isaac Sim)
    # -------------------------------
    if args.task:
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        env_cfg, agent_cfg, source_label = _load_from_task(
            args.task, args.agent_entry
        )

    # -------------------------------
    # LOG MODE (no simulator needed)
    # -------------------------------
    else:
        run_dir = _resolve_run_dir(args.run_dir, args.experiment)

        env_cfg = _load_yaml(run_dir / "params" / "env.yaml")
        agent_cfg = _load_yaml(run_dir / "params" / "agent.yaml")
        source_label = str(run_dir)

    # -------------------------------
    # BUILD REPORT
    # -------------------------------
    report = build_report(env_cfg, agent_cfg, source_label)
    print(report)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(report)

    # -------------------------------
    # CLEANUP
    # -------------------------------
    if simulation_app is not None:
        simulation_app.close()

if __name__ == "__main__":
    main()
