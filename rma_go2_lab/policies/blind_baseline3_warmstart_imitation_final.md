# Blind Baseline 3 Freeze

This file freezes the final definition of Baseline 3 for the current project
phase.

Baseline 3 is now frozen. Do not retune or overwrite it in place. Any future
changes should create a new explicitly versioned baseline.

## Identity

- canonical name: `blind_baseline3_warmstart_imitation_final`
- checkpoint:
  - `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_baseline_rough_warmstart_imitation/2026-04-18_12-30-41`
- selected source checkpoint:
  - `model_560.pt`

## Purpose

Baseline 3 is the frozen blind warm-start + imitation baseline.

It represents:

- deployable proprio-only control
- blind rough locomotion
- actor warm-started from the selected flat prior
- temporary flat-prior imitation regularization during early training
- no privileged information
- no latent `z`
- no adaptation module

It is used to answer:

- whether temporary imitation from the flat prior improves over warm-start
  alone
- how much behavior structure can be preserved while adapting to rough terrain

## Training Definition

- task:
  - `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation`
- env config:
  - `rma_go2_lab/envs/blind/rough_cfg.py`
- PPO config:
  - `rma_go2_lab/models/blind/variants_ppo_cfg.py`
- flat prior used for warm-start and imitation:
  - `rma_go2_lab/policies/flat1499.pt`

Frozen B3 characteristics retained in this version:

- same warm-start backbone as Baseline 2
- temporary imitation prior from the flat actor only
- softened imitation schedule:
  - `flat_imitation_coef_stage0 = 0.1`
  - `flat_imitation_coef_stage1 = 0.03`
  - `flat_imitation_stage0_end = 150`
  - `flat_imitation_stage1_end = 400`
- corrected flat-prior lineage and corrected frozen-expert loading were both in
  place before this selected run

Shared training terrain regime retained in this frozen version:

- mixed rough-terrain curriculum from `rma_go2_lab/envs/blind/rough_cfg.py`
- `terrain_levels` curriculum enabled
- `max_init_terrain_level = 2`
- stair-focused sub-terrains disabled:
  - `pyramid_stairs = 0.0`
  - `pyramid_stairs_inv = 0.0`
- box obstacles disabled:
  - `boxes = 0.0`
- retained rough-terrain families include:
  - `random_rough = 0.2`
  - `hf_pyramid_slope = 0.1`
  - `hf_pyramid_slope_inv = 0.1`

## Selection Rationale

Baseline 3 was not frozen from the final checkpoint by default.

The run was screened at multiple saved checkpoints:

- `model_560.pt`
- `model_960.pt`
- `model_1400.pt`
- `model_1999.pt`

`model_560.pt` was selected because it gave the best overall balance of:

- forward gait quality
- suite robustness balance
- nominal rough-terrain performance

The final checkpoint `model_1999.pt` looked stronger on some scalar suite
metrics, but behaviorally it had drifted toward a worse forward gait and a more
base-contact-dominated failure profile.

## Final Evaluation Artifacts

Controller-quality checks:

- `artifacts/evaluations/baseline3/gait_blind_warmstart_imitation_model560_standstill.json`
- `artifacts/evaluations/baseline3/gait_blind_warmstart_imitation_model560_forward.json`

Frozen mismatch suite:

- `artifacts/evaluations/baseline3/isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline3/isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`

Canonical clip set:

- `artifacts/evaluations/clips/baseline3/model560_nominal_overview_20260420_102917/`
- `artifacts/evaluations/clips/baseline3/model560_nominal_hero_20260420_102922/`
- `artifacts/evaluations/clips/baseline3/model560_low_friction_hero_20260420_102927/`
- `artifacts/evaluations/clips/baseline3/model560_weak_motor_hero_20260420_102935/`
- `artifacts/evaluations/clips/baseline3/model560_heavy_mass_hero_20260420_102940/`

## Final Behavior Summary

Training-side regime near the selected checkpoint window:

- reward was already in the strong mature range
- terrain curriculum was near the top of the shared rough regime
- imitation had already finished shaping the policy and was no longer needed to
  sustain behavior

Standstill eval:

- standstill planar speed:
  - `0.0284`
- standstill foot speed:
  - `0.0545`
- standstill joint velocity abs mean:
  - `0.2701`

Forward eval:

- achieved planar speed:
  - `0.3677`
- commanded planar speed:
  - `0.7492`
- diagonal trot score:
  - `0.1691`
- forward lateral drift per meter:
  - `2.3628`
- forward base tilt proxy:
  - `0.1836`
- gait interpretation:
  - `non_trot_or_serial`

Suite summary:

- average velocity error across the frozen suite:
  - `0.1368`
- average timeout fraction across the frozen suite:
  - `0.2990`
- average base-contact fraction across the frozen suite:
  - `0.7010`

Selected scenario reads:

- nominal random rough level 5:
  - `vel_err_step_mean = 0.1290`
  - `timeout_fraction_of_terminals = 0.1971`
  - `base_contact_fraction_of_terminals = 0.8029`
- low friction random rough level 5:
  - `base_contact_fraction_of_terminals = 0.9559`
- heavy mass random rough level 5:
  - `base_contact_fraction_of_terminals = 0.9358`
- weak motors random rough level 5:
  - `base_contact_fraction_of_terminals = 0.9792`

## Frozen Failure Mode Summary

Baseline 3 improves over scratch and preserves more useful locomotion structure
than the weakest B3 checkpoints, but it does not become a clean trot-specialist
or eliminate base-contact failures under mismatch.

Persistent pattern:

- forward gait remains mixed rather than cleanly diagonal
- weak motors and low friction remain the clearest stressors
- base-contact failures still dominate the harshest mismatch cases

Interpretation:

- the temporary imitation prior helped shape a stronger mid-run controller
- but the best B3 checkpoint is still best understood as a strong exploratory
  baseline, not an unquestioned replacement for Baseline 2 on every metric

## Benchmark Role

This baseline is the completed third rung in the frozen blind comparison
ladder:

- Baseline 1:
  - scratch
- Baseline 2:
  - warm-start
- Baseline 3:
  - warm-start + temporary imitation

It should be discussed as:

- the best B3 checkpoint found from a checkpoint sweep
- not simply the last checkpoint of the run

## Freeze Statement

Baseline 3 is now frozen.

Do not change:

- its checkpoint identity
- its checkpoint-sweep selection logic
- its warm-start source
- its imitation schedule
- its evaluation suite

If a stronger imitation baseline is trained later, it must be created as an
explicit new version rather than silently replacing this one.
