"""Run isolated evaluation suites as separate IsaacLab processes.

This is intentionally a process-level orchestrator, not an IsaacLab app script.
Creating and destroying multiple ManagerBasedRLEnv instances inside a single
SimulationApp can hang in PhysX/Kit for some terrain setups. Running each
scenario in a fresh IsaacLab process is slower, but much more reliable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ISAACLAB = Path("/home/bhuvan/tools/IsaacLab/isaaclab.sh")
REPO_ROOT = Path(__file__).resolve().parents[2]
ISOLATED_EVAL = REPO_ROOT / "scripts/eval/isolated.py"
DEFAULT_CHECKPOINT = Path(
    "/home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt"
)
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


@dataclass(frozen=True)
class Scenario:
    name: str
    terrain_type: str | None = None
    terrain_level: int | None = None
    latent_mode: str | None = None
    static_friction: float | None = None
    dynamic_friction: float | None = None
    mass_offset: float | None = None
    motor_stiffness_scale: float | None = None
    motor_damping_scale: float | None = None


def scenario_set(name: str) -> list[Scenario]:
    if name == "blind_baseline_v1":
        return [
            Scenario("nominal_random_rough_l5", terrain_type="random_rough", terrain_level=5),
            Scenario("nominal_slope_up_l5", terrain_type="hf_pyramid_slope", terrain_level=5),
            Scenario("nominal_slope_down_l5", terrain_type="hf_pyramid_slope_inv", terrain_level=5),
            Scenario("fric_min_random_rough_l5", terrain_type="random_rough", terrain_level=5, static_friction=0.1, dynamic_friction=0.1),
            Scenario("fric_max_random_rough_l5", terrain_type="random_rough", terrain_level=5, static_friction=2.0, dynamic_friction=2.0),
            Scenario("mass_min_random_rough_l5", terrain_type="random_rough", terrain_level=5, mass_offset=-2.0),
            Scenario("mass_max_random_rough_l5", terrain_type="random_rough", terrain_level=5, mass_offset=4.0),
            Scenario("motor_min_random_rough_l5", terrain_type="random_rough", terrain_level=5, motor_stiffness_scale=0.6, motor_damping_scale=0.6),
            Scenario("motor_max_random_rough_l5", terrain_type="random_rough", terrain_level=5, motor_stiffness_scale=1.4, motor_damping_scale=1.4),
        ]
    if name == "friction_only":
        return [
            Scenario("fric_0p1", static_friction=0.1, dynamic_friction=0.1),
            Scenario("fric_0p5", static_friction=0.5, dynamic_friction=0.5),
            Scenario("fric_1p0", static_friction=1.0, dynamic_friction=1.0),
            Scenario("fric_2p0", static_friction=2.0, dynamic_friction=2.0),
        ]
    if name == "mass_only":
        return [
            Scenario("mass_m2", mass_offset=-2.0),
            Scenario("mass_0", mass_offset=0.0),
            Scenario("mass_p2", mass_offset=2.0),
            Scenario("mass_p4", mass_offset=4.0),
        ]
    if name == "motor_only":
        return [
            Scenario("motor_0p6", motor_stiffness_scale=0.6, motor_damping_scale=0.6),
            Scenario("motor_0p8", motor_stiffness_scale=0.8, motor_damping_scale=0.8),
            Scenario("motor_1p0", motor_stiffness_scale=1.0, motor_damping_scale=1.0),
            Scenario("motor_1p4", motor_stiffness_scale=1.4, motor_damping_scale=1.4),
        ]
    if name == "terrain_only":
        return [
            Scenario("random_rough_l9", terrain_type="random_rough", terrain_level=9),
            Scenario("stairs_up_l9", terrain_type="pyramid_stairs", terrain_level=9),
            Scenario("stairs_down_l9", terrain_type="pyramid_stairs_inv", terrain_level=9),
            Scenario("boxes_l9", terrain_type="boxes", terrain_level=9),
        ]
    return [
        Scenario("baseline_nominal"),
        Scenario("fric_0p1", static_friction=0.1, dynamic_friction=0.1),
        Scenario("fric_0p5", static_friction=0.5, dynamic_friction=0.5),
        Scenario("fric_2p0", static_friction=2.0, dynamic_friction=2.0),
        Scenario("mass_p2", mass_offset=2.0),
        Scenario("mass_p4", mass_offset=4.0),
        Scenario("motor_0p8", motor_stiffness_scale=0.8, motor_damping_scale=0.8),
        Scenario("motor_1p4", motor_stiffness_scale=1.4, motor_damping_scale=1.4),
    ]


def add_override_args(cmd: list[str], scenario: Scenario) -> None:
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


def flatten_row(result: dict) -> dict:
    row = {
        "scenario": result["scenario"],
        "score": result["score"],
        "returncode": result.get("returncode", 0),
        "terrain_type": result.get("terrain_type"),
        "terrain_level": result.get("terrain_level"),
        "latent_mode": result.get("latent_mode"),
    }
    row.update(result.get("overrides", {}))
    row.update(result.get("metrics", {}))
    row.update(result.get("constraint_checks", {}))
    row.update({f"valid_count_{key}": value for key, value in result.get("valid_counts", {}).items()})
    return row


def write_outputs(results: list[dict], args: argparse.Namespace, total_scenarios: int) -> None:
    rows = sorted(results, key=lambda row: row.get("score", float("-inf")), reverse=True)
    out = {
        "status": "complete"
        if len(results) == total_scenarios and all(r.get("returncode", 0) == 0 for r in results)
        else "partial",
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "suite": args.suite,
        "scenario_count": len(results),
        "expected_scenario_count": total_scenarios,
        "terrain_type": args.terrain_type,
        "terrain_level": args.terrain_level,
        "latent_mode": args.latent_mode,
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
    parser = argparse.ArgumentParser(description="Run isolated eval scenarios in separate IsaacLab processes.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task", default="RMA-Go2-Blind-Baseline-Rough-WarmStart")
    parser.add_argument("--terrain-type", default="random_rough")
    parser.add_argument("--terrain-level", type=int, default=-1, help="Fixed terrain curriculum level. Use -1 for spread up to max_init_terrain_level.")
    parser.add_argument("--latent-mode", default="normal", choices=LATENT_MODES)
    parser.add_argument(
        "--suite",
        default="friction_only",
        choices=[
            "default",
            "blind_baseline_v1",
            "friction_only",
            "mass_only",
            "motor_only",
            "terrain_only",
        ],
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/evaluations")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--continue-on-error", action="store_true", default=False)
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    else:
        args.output_dir = args.output_dir.resolve()

    level_part = f"level{args.terrain_level}" if args.terrain_level >= 0 else "levelspread"
    stem = f"isolated_suite_{args.checkpoint.stem}_{args.suite}_{args.terrain_type}_{level_part}_{args.latent_mode}_seed{args.seed}"
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
    tmp_dir = args.output_dir / "tmp" / args.json_out.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    scenarios = scenario_set(args.suite)
    for idx, scenario in enumerate(scenarios, start=1):
        scenario_json = tmp_dir / f"{idx:02d}_{scenario.name}.json"
        terrain_type = scenario.terrain_type or args.terrain_type
        terrain_level = args.terrain_level if scenario.terrain_level is None else scenario.terrain_level
        latent_mode = scenario.latent_mode or args.latent_mode
        cmd = [
            str(ISAACLAB),
            "-p",
            str(ISOLATED_EVAL),
            "--checkpoint",
            str(args.checkpoint),
            "--task",
            args.task,
            "--terrain-type",
            terrain_type,
            "--latent-mode",
            latent_mode,
            "--terrain-level",
            str(terrain_level),
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--seed",
            str(args.seed),
            "--json-out",
            str(scenario_json),
        ]
        if args.headless:
            cmd.append("--headless")
        add_override_args(cmd, scenario)

        print(f"[{idx}/{len(scenarios)}] {scenario.name}: {' '.join(cmd)}", flush=True)
        child_env = os.environ.copy()
        child_env["TERM"] = child_env.get("TERM") if child_env.get("TERM") not in (None, "", "dumb") else "xterm"
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=child_env)
        if scenario_json.exists():
            result = json.loads(scenario_json.read_text())
            result["scenario"] = scenario.name
            result["returncode"] = proc.returncode
        else:
            result = {
                "scenario": scenario.name,
                "score": float("-inf"),
                "returncode": proc.returncode,
                "terrain_type": terrain_type,
                "terrain_level": terrain_level,
                "latent_mode": latent_mode,
                "overrides": scenario.__dict__,
                "metrics": {},
            }
        results.append(result)
        write_outputs(results, args, len(scenarios))

        if proc.returncode != 0 and not args.continue_on_error:
            print(f"[ERROR] {scenario.name} failed with return code {proc.returncode}", file=sys.stderr)
            return proc.returncode

    print(f"[INFO] Wrote JSON: {args.json_out}")
    print(f"[INFO] Wrote CSV:  {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
