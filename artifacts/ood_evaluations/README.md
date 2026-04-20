# OOD Evaluation Artifacts

This directory is for exploratory out-of-distribution evaluation only.

It is intentionally separate from:

- `artifacts/evaluations/`

Use this path when probing the limits of frozen baselines under:

- unseen terrain families
- stronger geometry
- stronger dynamics mismatch
- longer or harsher evaluation conditions

## Layout

Recommended subdirectories:

- `artifacts/ood_evaluations/baseline1/`
- `artifacts/ood_evaluations/baseline2/`
- `artifacts/ood_evaluations/baseline3/`
- `artifacts/ood_evaluations/clips/`

Recommended clip subdirectories:

- `artifacts/ood_evaluations/clips/baseline1/`
- `artifacts/ood_evaluations/clips/baseline2/`
- `artifacts/ood_evaluations/clips/baseline3/`

Per-baseline `_tmp/` subdirectories are expected.

They hold per-scenario intermediate JSONs written by `run_ood_suite.py` before
the final consolidated suite JSON and CSV are assembled. Treat them as
exploratory intermediate state, not as the main result files.

## Important Rule

Do not mix OOD exploratory results with canonical baseline-comparison artifacts.

Canonical baseline comparison stays in:

- `artifacts/evaluations/`

If an OOD probe later becomes important enough to promote, do that as an
explicit later decision rather than by writing it here first and quietly
relabeling it later.

## Current Exploratory Runner

Use:

- `scripts/eval_ood/run_ood_suite.py`

Current named suites:

- `ood_geometry_v1`
- `ood_dynamics_v1`
- `ood_combo_v1`
- `ood_push_v1`
- `ood_switch_v1`
- `ood_limit_v1`

## Current Findings Note

The first written comparison of warm-start OOD behavior lives in:

- `docs/OOD_FINDINGS_B1_B2_B3.md`

That note is exploratory and does not replace the canonical baseline
comparison.
