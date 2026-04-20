# Blind Baseline 1 Freeze

This file freezes the final definition of Baseline 1 for the current project
phase.

Baseline 1 is now frozen. Do not retune or overwrite it in place. Any future
changes should create a new explicitly versioned baseline.

## Identity

- canonical name: `blind_baseline1_scratch_final`
- checkpoint: `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_baseline_rough_scratch/2026-04-17_16-46-29`
- selected source checkpoint:
  - `model_1999.pt`

## Purpose

Baseline 1 is the frozen blind scratch baseline.

It represents:

- deployable proprio-only control
- blind rough locomotion
- no warm-start initialization
- no privileged information
- no latent `z`
- no adaptation module

It is used to answer:

- what a scratch-trained blind rough controller looks like under the frozen SOP
- how much warm-starting from the selected flat prior helps beyond scratch

## Training Definition

- task:
  - `RMA-Go2-Blind-Baseline-Rough`
- env config:
  - `rma_go2_lab/envs/blind/rough_cfg.py`
- PPO config:
  - `rma_go2_lab/models/blind/variants_ppo_cfg.py`

Frozen characteristics retained in this version:

- scratch actor initialization
- `init_noise_std = 0.35`
- `entropy_coef = 0.002`
- `desired_kl = 0.01`
- no flat-prior warm-start

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

`model_1999.pt` was selected because:

- training reached the intended 2000-iteration horizon
- the run produced functional forward rough locomotion without collapse
- the resulting gait and failure modes remained interpretable as a scratch
  baseline

This baseline is not frozen because it is pretty. It is frozen because it is a
fair and informative scratch comparator under the same rough-task recipe used by
the later baselines.

## Final Evaluation Artifacts

Controller-quality checks:

- `artifacts/evaluations/baseline1/gait_blind_scratch_model1999_standstill.json`
- `artifacts/evaluations/baseline1/gait_blind_scratch_model1999_forward.json`

Frozen mismatch suite:

- `artifacts/evaluations/baseline1/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline1/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`

Frozen clip set:

- `artifacts/evaluations/clips/baseline1/nominal_overview_20260418_100751/`
- `artifacts/evaluations/clips/baseline1/nominal_hero_20260418_100756/`
- `artifacts/evaluations/clips/baseline1/low_friction_hero_20260418_100804/`
- `artifacts/evaluations/clips/baseline1/weak_motor_hero_20260418_100811/`
- `artifacts/evaluations/clips/baseline1/heavy_mass_hero_20260418_100816/`

## Final Behavior Summary

Training-side final regime near freeze:

- tracking error:
  - `error_vel_xy = 0.1380`
  - `error_vel_yaw = 0.2262`
- training terrain regime:
  - shared mixed rough curriculum from `rough_cfg.py`
- curriculum hardness:
  - final `Curriculum/terrain_levels = 4.5541`
- terminations:
  - `time_out = 0.8160`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0031`
  - `base_height = 0.1662`
  - `low_progress = 0.0165`

Standstill eval:

- standstill planar speed:
  - `0.0250`
- standstill foot speed:
  - `0.0594`
- standstill joint velocity abs mean:
  - `0.2863`
- standstill base tilt proxy:
  - `0.0389`

Forward eval:

- achieved planar speed:
  - `0.3506`
- commanded planar speed:
  - `0.7497`
- forward lateral drift per meter:
  - `2.5436`
- forward base tilt proxy:
  - `0.2539`
- foot slip contact mean:
  - `0.1499`
- diagonal trot score:
  - `0.0416`
- diagonal exact-pair swing fraction:
  - `0.0047`
- fore-hind exact-pair swing fraction:
  - `0.1997`
- gait interpretation:
  - `non_trot_or_serial`

## Frozen Failure Mode Summary

The dominant limitation is not optimizer instability. It is the quality of the
learned rough locomotion strategy under hidden mismatch.

Persistent pattern:

- locomotion is functional but behaviorally coarse
- forward motion remains bound-prone rather than diagonal-trot-like
- forward eval terminations are fully base-contact dominated
- the run climbed into mid-to-high rough curriculum, but did not reach the same
  final mean terrain level as the warm-start baseline
- mismatch degradation remains sharp under low friction, weak motors, and heavy
  mass

In the `blind_baseline_v1` suite, the strongest degradation appears under:

- low friction
- weak motors
- heavy mass

The strongest suite case is:

- `motor_max_random_rough_l5`

Interpretation note:

- the suite score ordering is not the same thing as “best behavior quality”
- the gait read and failure signatures still matter more than score alone

## Benchmark Role

This baseline is the fixed scratch comparison point for:

- Baseline 2:
  - warm-started blind baseline
- Baseline 3:
  - warm-start + imitation blind baseline

Interpretation rule:

- improvements over this baseline should be discussed in terms of tracking,
  failure robustness, mismatch generalization, control effort, and behavior
  quality
- not reward alone

## Freeze Statement

Baseline 1 is now frozen.

Do not change:

- its checkpoint identity
- its scratch training configuration
- its evaluation suite
- its canonical clip set

If a stronger scratch blind baseline is trained later, it must be created as an
explicit new baseline version rather than silently replacing this one.
