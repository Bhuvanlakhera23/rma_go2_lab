"""Plot consolidated isolated evaluation suite results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, "")
        return default if value == "" else float(value)
    except Exception:
        return default


def _short_name(name: str) -> str:
    replacements = {
        "nominal_": "",
        "_random_rough_l9": "",
        "_stairs_up_l9": "_up",
        "_stairs_down_l9": "_down",
        "_l9": "",
        "combined_": "combo_",
        "high_grip_heavy_strong": "hi-grip_heavy_strong",
        "low_grip_heavy_weak": "lo-grip_heavy_weak",
        "latent_shuffled": "z_shuf",
        "latent_zero": "z_zero",
        "random_rough": "rough",
        "pyramid_stairs_inv": "stairs_down",
        "pyramid_stairs": "stairs_up",
    }
    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_survival(rows: list[dict[str, str]], out_dir: Path, stem: str) -> Path:
    labels = [_short_name(row["scenario"]) for row in rows]
    timeout_rates = [_float(row, "timeout_events_per_env") for row in rows]
    contact_rates = [_float(row, "base_contact_events_per_env") for row in rows]

    fig, ax = plt.subplots(figsize=(13, 7))
    y = range(len(rows))
    ax.barh(y, timeout_rates, color="#2d9c7f", label="timeouts per env")
    ax.barh(y, contact_rates, left=timeout_rates, color="#d65a4a", label="base contacts per env")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.8, max(t + c for t, c in zip(timeout_rates, contact_rates)) * 1.08))
    ax.set_xlabel("events per env over rollout")
    ax.set_title("Teacher Stress Suite: Survival vs Base-Contact Failures")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    path = out_dir / f"{stem}_survival.png"
    _save(fig, path)
    return path


def plot_tracking(rows: list[dict[str, str]], out_dir: Path, stem: str) -> Path:
    labels = [_short_name(row["scenario"]) for row in rows]
    vel_err = [_float(row, "vel_err_step_mean") for row in rows]
    yaw_err = [_float(row, "yaw_err_step_mean") for row in rows]

    fig, ax = plt.subplots(figsize=(13, 7))
    y = range(len(rows))
    ax.barh([v - 0.18 for v in y], vel_err, height=0.35, color="#3b75af", label="linear velocity error")
    ax.barh([v + 0.18 for v in y], yaw_err, height=0.35, color="#d49a3a", label="yaw velocity error")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("step mean error")
    ax.set_title("Teacher Stress Suite: Tracking Errors")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    path = out_dir / f"{stem}_tracking.png"
    _save(fig, path)
    return path


def plot_scorecard(rows: list[dict[str, str]], out_dir: Path, stem: str) -> Path:
    metrics = [
        ("score", "Score", False),
        ("timeout_events_per_env", "Timeouts/env", False),
        ("base_contact_events_per_env", "Contacts/env", True),
        ("vel_err_step_mean", "Vel err", True),
        ("yaw_err_step_mean", "Yaw err", True),
    ]
    labels = [_short_name(row["scenario"]) for row in rows]
    data: list[list[float]] = []
    for key, _, invert in metrics:
        values = [_float(row, key) for row in rows]
        lo, hi = min(values), max(values)
        span = hi - lo if hi != lo else 1.0
        normalized = [(v - lo) / span for v in values]
        if invert:
            normalized = [1.0 - v for v in normalized]
        data.append(normalized)

    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([name for _, name, _ in metrics])
    ax.set_title("Teacher Stress Suite Scorecard (Green = Better Within This Suite)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    path = out_dir / f"{stem}_scorecard.png"
    _save(fig, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot consolidated isolated suite CSV results.")
    parser.add_argument("csv", type=Path, help="Consolidated suite CSV.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to <csv parent>/plots.")
    args = parser.parse_args()

    rows = _load_rows(args.csv)
    out_dir = args.out_dir or args.csv.parent / "plots"
    stem = args.csv.stem

    outputs = [
        plot_survival(rows, out_dir, stem),
        plot_tracking(rows, out_dir, stem),
        plot_scorecard(rows, out_dir, stem),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
