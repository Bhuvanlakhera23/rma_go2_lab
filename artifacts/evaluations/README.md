# Evaluation Artifacts

This directory is the canonical place for checkpoint evaluation outputs.

Use per-artifact subdirectories rather than keeping all outputs at the top
level.

Current layout:

- `artifacts/evaluations/flat_prior/`
- `artifacts/evaluations/baseline1/`
- `artifacts/evaluations/baseline2/`
- `artifacts/evaluations/baseline3/`
- `artifacts/evaluations/clips/`

Clip layout:

- `artifacts/evaluations/clips/flat_prior/`
- `artifacts/evaluations/clips/baseline1/`
- `artifacts/evaluations/clips/baseline2/`
- `artifacts/evaluations/clips/baseline3/`

## Current evaluators

- `scripts/eval/isolated.py` writes one detailed JSON result by default into this directory.
- `scripts/eval/run_isolated_suite.py` is the canonical suite runner. It launches each scenario in a fresh IsaacLab process, then writes one consolidated JSON and CSV. This avoids hangs from repeatedly creating IsaacLab environments inside one SimulationApp.
- `scripts/eval/gait.py` is the controller-quality evaluator for standstill, forward drift, step response, and contact-derived gait structure.
- `scripts/eval/blind_baseline_diagnostics.py` summarizes reward and termination health for blind-baseline checkpoints.

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

## Canonical blind baseline suite

Use `run_isolated_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --suite blind_baseline_v1` for the frozen blind-baseline mismatch suite.

It includes:

- nominal random rough at fixed level 5
- fixed uphill and downhill slope checks at level 5
- friction min and max on fixed random rough
- mass min and max on fixed random rough
- motor-strength min and max on fixed random rough

Use `gait.py` in addition to the suite:

- `--command-profile standstill` for quiet-stop quality
- `--command-profile forward` for forward tracking, drift, and gait structure

The suite is intentionally conservative. It is meant to expose interpretable degradation in a fixed blind controller, not to maximize obstacle difficulty.

## Canonical Baseline Videos

Frozen blind baselines should also carry a small fixed clip set in addition to
JSON and CSV evaluation outputs.

Current recording helpers:

- `scripts/eval/record_clip.py`
- `scripts/eval/record_overview_clip.py`

The frozen clip manifest is defined in:

- `rma_go2_lab/policies/blind_baseline_protocol.md`

Use the same clip conditions across B1, B2, and B3. Only the checkpoint should
change when comparing baselines.

Store clips under the artifact-specific clip directory rather than directly
under `artifacts/evaluations/clips/`.

Each recorded clip directory should contain:

- one `.mp4`
- one `metadata.json` describing the checkpoint, tag, terrain settings, and
  fixed override values used for that recording

## Canonical flat-prior sanity reports

The selected flat prior is not a full benchmark target. It only needs a small
sanity record as a reusable locomotion prior.

Canonical reports:

- `artifacts/evaluations/flat_prior/gait_flat_prior_model1500_standstill.json`
- `artifacts/evaluations/flat_prior/gait_flat_prior_model1500_forward.json`

Treat older auto-named flat gait files as stale unless they are explicitly
referenced by a manifest.
