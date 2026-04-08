# Evaluation Artifacts

This directory is the canonical place for checkpoint evaluation outputs.

## Current evaluators

- `scripts/eval_isolated.py` writes one detailed JSON result by default into this directory.
- `scripts/run_isolated_suite.py` is the canonical suite runner. It launches each scenario in a fresh IsaacLab process, then writes one consolidated JSON and CSV. This avoids hangs from repeatedly creating IsaacLab environments inside one SimulationApp.

## Metric semantics

Continuous quantities are reported as step means:

- `reward_step_mean`
- `vel_err_step_mean`
- `yaw_err_step_mean`
- `terrain_level_step_mean`

Termination quantities are counted from `dones` and `infos["time_outs"]`:

- `terminal_dones`
- `terminal_timeouts`
- `terminal_base_contacts`
- `timeout_fraction_of_terminals`
- `base_contact_fraction_of_terminals`
- `timeout_events_per_env`
- `base_contact_events_per_env`

Do not compare new terminal metrics directly to old root-level JSON files from earlier evaluator versions. Those old files used different aggregation semantics and should be treated as obsolete provenance, not qualification evidence.

## Fixed terrain levels

When `--terrain-level` is set, terrain curriculum is disabled and the evaluator forces all environments to that exact level. In that mode IsaacLab no longer logs `Curriculum/terrain_levels`, so `terrain_level_source` is reported as `forced`.

## Complete teacher suite

Use `run_isolated_suite.py --suite complete_teacher` for the broad teacher stress test. It includes:

- fixed level-9 terrain family checks: random rough, stairs up, stairs down, and boxes
- isolated training-randomization checks: friction min/max, mass min/max, and motor gain min/max
- combined stress cases that stack hard terrain with extreme dynamics
- latent ablations: normal vs. zeroed/shuffled latent on rough and stairs

The suite also records `constraint_checks` fields such as observed friction, mass offset, and motor stiffness scale when IsaacLab exposes them, so the output can catch override no-ops instead of only recording requested values.
