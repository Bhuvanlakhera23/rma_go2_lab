"""Run exploratory OOD evaluation suites as separate IsaacLab processes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from ood_scenarios import OODScenario, scenario_set


ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_EVAL = REPO_ROOT / "scripts/eval/isolated.py"
DEFAULT_CHECKPOINT = REPO_ROOT / "rma_go2_lab/policies/blind_baseline2_warmstart_final.pt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/ood_evaluations"
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


def add_override_args(cmd: list[str], scenario: OODScenario) -> None:
    if scenario.static_friction is not None:
        cmd += ["--static-friction", str(scenario.static_friction)]
    if scenario.dynamic_friction is not None:
        cmd += ["--dynamic-friction", str(scenario.dynamic_friction)]
    if scenario.mass_offset is not None:
        cmd += ["--mass-offset", str(scenario.mass_offset)]
    if scenario.motor_stiffness_scale is not None:
        cmd += ["--motor-stiffness-scale", str(scenario.motor_stiffness_scale)]
    if scenario.motor_damping_scale is not None:
        cmd += ["--motor-damping-scale", str(scenario.motor_damping_scale)]
    if scenario.push_interval_s is not None:
        cmd += ["--push-interval-min-s", str(scenario.push_interval_s[0])]
        cmd += ["--push-interval-max-s", str(scenario.push_interval_s[1])]
    if scenario.push_velocity_range is not None:
        axis_to_flag = {
            "x": "push-x-range",
            "y": "push-y-range",
            "z": "push-z-range",
            "roll": "push-roll-range",
            "pitch": "push-pitch-range",
            "yaw": "push-yaw-range",
        }
        for axis, value_range in scenario.push_velocity_range.items():
            cmd += [f"--{axis_to_flag[axis]}={value_range[0]},{value_range[1]}"]
    if scenario.switch_step is not None:
        cmd += ["--switch-step", str(scenario.switch_step)]
    if scenario.switch_static_friction is not None:
        cmd += ["--switch-static-friction", str(scenario.switch_static_friction)]
    if scenario.switch_dynamic_friction is not None:
        cmd += ["--switch-dynamic-friction", str(scenario.switch_dynamic_friction)]
    if scenario.switch_mass_offset is not None:
        cmd += ["--switch-mass-offset", str(scenario.switch_mass_offset)]
    if scenario.switch_motor_stiffness_scale is not None:
        cmd += ["--switch-motor-stiffness-scale", str(scenario.switch_motor_stiffness_scale)]
    if scenario.switch_motor_damping_scale is not None:
        cmd += ["--switch-motor-damping-scale", str(scenario.switch_motor_damping_scale)]


def flatten_row(result: dict) -> dict:
    row = {
        "rank": result.get("rank"),
        "scenario": result["scenario"],
        "score": result["score"],
        "returncode": result.get("returncode", 0),
        "terrain_type": result.get("terrain_type"),
        "terrain_level": result.get("terrain_level"),
        "latent_mode": result.get("latent_mode"),
    }
    row.update(result.get("overrides", {}))
    row.update(result.get("metrics", {}))
    row.update({f"post_switch_{k}": v for k, v in result.get("post_switch_metrics", {}).items()})
    row.update(result.get("constraint_checks", {}))
    row.update({f"post_switch_{k}": v for k, v in result.get("post_switch_constraint_checks", {}).items()})
    row.update({f"valid_count_{k}": v for k, v in result.get("valid_counts", {}).items()})
    return row


def write_outputs(results: list[dict], args: argparse.Namespace, total_scenarios: int) -> None:
    rows = sorted(results, key=lambda row: row.get("score", float("-inf")), reverse=True)
    out = {
        "status": "complete"
        if len(results) == total_scenarios and all(r.get("returncode", 0) == 0 for r in results)
        else "partial",
        "mode": "exploratory_ood",
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "suite": args.suite,
        "scenario_count": len(results),
        "expected_scenario_count": total_scenarios,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "results": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    fieldnames = [
        "rank",
        "scenario",
        "score",
        "returncode",
        "terrain_type",
        "terrain_level",
        "latent_mode",
        "reward_step_mean",
        "vel_err_step_mean",
        "yaw_err_step_mean",
        "terrain_level_step_mean",
        "terrain_level_source",
        "terminal_dones",
        "terminal_timeouts",
        "terminal_base_contacts",
        "timeout_fraction_of_terminals",
        "base_contact_fraction_of_terminals",
        "timeout_events_per_env",
        "base_contact_events_per_env",
        "static_friction",
        "dynamic_friction",
        "mass_offset",
        "motor_stiffness_scale",
        "motor_damping_scale",
        "switch_step",
        "switch_static_friction",
        "switch_dynamic_friction",
        "switch_mass_offset",
        "switch_motor_stiffness_scale",
        "switch_motor_damping_scale",
        "push_interval_min_s",
        "push_interval_max_s",
        "push_x_min",
        "push_x_max",
        "push_y_min",
        "push_y_max",
        "push_z_min",
        "push_z_max",
        "push_roll_min",
        "push_roll_max",
        "push_pitch_min",
        "push_pitch_max",
        "push_yaw_min",
        "push_yaw_max",
        "observed_static_friction_mean",
        "observed_static_friction_min",
        "observed_static_friction_max",
        "observed_dynamic_friction_mean",
        "observed_dynamic_friction_min",
        "observed_dynamic_friction_max",
        "observed_mass_offset_mean",
        "observed_mass_offset_min",
        "observed_mass_offset_max",
        "observed_motor_stiffness_scale_mean",
        "observed_motor_stiffness_scale_min",
        "observed_motor_stiffness_scale_max",
        "observed_push_interval_min_s",
        "observed_push_interval_max_s",
        "post_switch_observed_static_friction_mean",
        "post_switch_observed_static_friction_min",
        "post_switch_observed_static_friction_max",
        "post_switch_observed_dynamic_friction_mean",
        "post_switch_observed_dynamic_friction_min",
        "post_switch_observed_dynamic_friction_max",
        "post_switch_observed_mass_offset_mean",
        "post_switch_observed_mass_offset_min",
        "post_switch_observed_mass_offset_max",
        "post_switch_observed_motor_stiffness_scale_mean",
        "post_switch_observed_motor_stiffness_scale_min",
        "post_switch_observed_motor_stiffness_scale_max",
        "post_switch_reward_step_mean",
        "post_switch_vel_err_step_mean",
        "post_switch_yaw_err_step_mean",
        "post_switch_terrain_level_step_mean",
        "post_switch_terminal_dones",
        "post_switch_terminal_timeouts",
        "post_switch_terminal_base_contacts",
        "post_switch_timeout_fraction_of_terminals",
        "post_switch_base_contact_fraction_of_terminals",
        "post_switch_timeout_events_per_env",
        "post_switch_base_contact_events_per_env",
        "valid_count_reward",
        "valid_count_vel_err",
        "valid_count_yaw_err",
        "valid_count_terrain_level",
    ]
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, result in enumerate(rows, start=1):
            row = flatten_row(result)
            row["rank"] = rank
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory OOD suites for frozen baselines.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default="RMA-Go2-Blind-Baseline-Rough-WarmStart")
    parser.add_argument(
        "--suite",
        default="ood_geometry_v1",
        choices=["ood_geometry_v1", "ood_dynamics_v1", "ood_combo_v1", "ood_push_v1", "ood_switch_v1", "ood_limit_v1"],
    )
    parser.add_argument("--latent-mode", default="normal", choices=LATENT_MODES)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    else:
        args.output_dir = args.output_dir.resolve()

    checkpoint_stem = args.checkpoint.stem
    stem = f"ood_suite_{checkpoint_stem}_{args.suite}_{args.latent_mode}_seed{args.seed}"
    if args.json_out is None:
        args.json_out = args.output_dir / f"{stem}.json"
    elif not args.json_out.is_absolute():
        args.json_out = REPO_ROOT / args.json_out
    else:
        args.json_out = args.json_out.resolve()

    if args.csv_out is None:
        args.csv_out = args.output_dir / f"{stem}.csv"
    elif not args.csv_out.is_absolute():
        args.csv_out = REPO_ROOT / args.csv_out
    else:
        args.csv_out = args.csv_out.resolve()

    return args


def main() -> int:
    args = parse_args()
    scenarios = scenario_set(args.suite)
    results: list[dict] = []

    for scenario in scenarios:
        json_tmp = args.output_dir / "_tmp" / f"{scenario.name}.json"
        json_tmp.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ISAACLAB),
            "-p",
            str(ISOLATED_EVAL),
            "--task",
            args.task,
            "--checkpoint",
            str(args.checkpoint),
            "--terrain-type",
            scenario.terrain_type or "random_rough",
            "--terrain-level",
            str(-1 if scenario.terrain_level is None else scenario.terrain_level),
            "--latent-mode",
            scenario.latent_mode or args.latent_mode,
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--json-out",
            str(json_tmp),
            "--headless",
        ]
        add_override_args(cmd, scenario)

        print(f"[OOD] Running {scenario.name}")
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if json_tmp.exists():
            result = json.loads(json_tmp.read_text())
            result["scenario"] = scenario.name
            result["returncode"] = proc.returncode
            results.append(result)
        else:
            results.append(
                {
                    "scenario": scenario.name,
                    "score": float("-inf"),
                    "returncode": proc.returncode,
                    "terrain_type": scenario.terrain_type,
                    "terrain_level": scenario.terrain_level,
                    "latent_mode": scenario.latent_mode or args.latent_mode,
                    "overrides": {
                        "static_friction": scenario.static_friction,
                        "dynamic_friction": scenario.dynamic_friction,
                        "mass_offset": scenario.mass_offset,
                        "motor_stiffness_scale": scenario.motor_stiffness_scale,
                        "motor_damping_scale": scenario.motor_damping_scale,
                        "switch_step": scenario.switch_step,
                        "switch_static_friction": scenario.switch_static_friction,
                        "switch_dynamic_friction": scenario.switch_dynamic_friction,
                        "switch_mass_offset": scenario.switch_mass_offset,
                        "switch_motor_stiffness_scale": scenario.switch_motor_stiffness_scale,
                        "switch_motor_damping_scale": scenario.switch_motor_damping_scale,
                        "push_interval_min_s": None if scenario.push_interval_s is None else scenario.push_interval_s[0],
                        "push_interval_max_s": None if scenario.push_interval_s is None else scenario.push_interval_s[1],
                    "push_x_min": None if scenario.push_velocity_range is None or "x" not in scenario.push_velocity_range else scenario.push_velocity_range["x"][0],
                    "push_x_max": None if scenario.push_velocity_range is None or "x" not in scenario.push_velocity_range else scenario.push_velocity_range["x"][1],
                    "push_y_min": None if scenario.push_velocity_range is None or "y" not in scenario.push_velocity_range else scenario.push_velocity_range["y"][0],
                    "push_y_max": None if scenario.push_velocity_range is None or "y" not in scenario.push_velocity_range else scenario.push_velocity_range["y"][1],
                    "push_z_min": None if scenario.push_velocity_range is None or "z" not in scenario.push_velocity_range else scenario.push_velocity_range["z"][0],
                    "push_z_max": None if scenario.push_velocity_range is None or "z" not in scenario.push_velocity_range else scenario.push_velocity_range["z"][1],
                    "push_roll_min": None if scenario.push_velocity_range is None or "roll" not in scenario.push_velocity_range else scenario.push_velocity_range["roll"][0],
                    "push_roll_max": None if scenario.push_velocity_range is None or "roll" not in scenario.push_velocity_range else scenario.push_velocity_range["roll"][1],
                    "push_pitch_min": None if scenario.push_velocity_range is None or "pitch" not in scenario.push_velocity_range else scenario.push_velocity_range["pitch"][0],
                    "push_pitch_max": None if scenario.push_velocity_range is None or "pitch" not in scenario.push_velocity_range else scenario.push_velocity_range["pitch"][1],
                    "push_yaw_min": None if scenario.push_velocity_range is None or "yaw" not in scenario.push_velocity_range else scenario.push_velocity_range["yaw"][0],
                    "push_yaw_max": None if scenario.push_velocity_range is None or "yaw" not in scenario.push_velocity_range else scenario.push_velocity_range["yaw"][1],
                },
                    "metrics": {},
                    "constraint_checks": {},
                    "valid_counts": {},
                }
            )
        if proc.returncode != 0 and not args.continue_on_error:
            print(f"[OOD] Stopping after failure in {scenario.name}")
            break

    write_outputs(results, args, total_scenarios=len(scenarios))
    print(f"[OOD] Wrote JSON to: {args.json_out}")
    print(f"[OOD] Wrote CSV  to: {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
