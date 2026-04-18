# Blind Baseline 2 Freeze

This file freezes the final definition of Baseline 2 for the current project
phase.

Baseline 2 is now frozen. Do not retune or overwrite it in place. Any future
changes should create a new explicitly versioned baseline.

## Identity

- canonical name: `blind_baseline2_warmstart_final`
- checkpoint: `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_blind_baseline_rough_warmstart/2026-04-16_12-38-11`
- selected source checkpoint:
  - `model_1500.pt`

## Purpose

Baseline 2 is the frozen blind warm-start baseline.

It represents:

- deployable proprio-only control
- blind rough locomotion
- actor warm-started from the selected flat prior
- no privileged information
- no latent `z`
- no adaptation module

It is used to answer:

- how much a flat locomotion prior helps blind rough locomotion
- what failure modes remain under hidden mismatch without adaptation

## Training Definition

- task:
  - `RMA-Go2-Blind-Baseline-Rough-WarmStart`
- env config:
  - `rma_go2_lab/envs/blind/rough_cfg.py`
- PPO config:
  - `rma_go2_lab/models/blind/variants_ppo_cfg.py`
- flat prior used for warm-start:
  - `rma_go2_lab/policies/flat1499.pt`

Final stabilization changes retained in this frozen version:

- lower warm-start exploration pressure
  - `init_noise_std = 0.35`
  - `entropy_coef = 0.002`
  - `desired_kl = 0.01`
- slightly stronger posture control
  - `flat_orientation_l2 = -1.0`
  - `ang_vel_xy_l2 = -0.075`

## Selection Rationale

`model_1500.pt` was selected because:

- training remained numerically stable deep into the run
- metrics had entered a mature, steady regime
- the resulting failure mode remained interpretable

This was treated as a stronger final choice than earlier mid-run candidates such
as `model_700.pt` or `model_1320.pt`.

## Final Evaluation Artifacts

Controller-quality checks:

- `artifacts/evaluations/baseline2/gait_blind_warmstart_model1500_standstill.json`
- `artifacts/evaluations/baseline2/gait_blind_warmstart_model1500_forward.json`

Frozen mismatch suite:

- `artifacts/evaluations/baseline2/isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline2/isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`

## Final Behavior Summary

Training-side final regime near freeze:

- tracking error:
  - `error_vel_xy = 0.1275`
  - `error_vel_yaw = 0.1930`
- terminations:
  - `time_out = 0.8242`
  - `base_contact = 0.0000`
  - `base_orientation = 0.0116`
  - `base_height = 0.1551`
  - `low_progress = 0.0098`

Standstill eval:

- standstill planar speed:
  - `0.0259`
- standstill foot speed:
  - `0.0558`
- standstill joint velocity abs mean:
  - `0.2619`

Forward eval:

- achieved planar speed:
  - `0.3749`
- commanded planar speed:
  - `0.7490`
- forward lateral drift per meter:
  - `1.6319`
- forward base tilt proxy:
  - `0.1573`
- gait interpretation:
  - `non_trot_or_serial`

## Frozen Failure Mode Summary

The dominant limitation is not PPO instability. It is terrain negotiation under
hidden mismatch.

Persistent pattern:

- most non-timeout failures still appear as base-contact failures
- rough forward locomotion remains awkward rather than cleanly trotting
- the main training-side bottleneck remains body-height / clearance loss

In the `blind_baseline_v1` suite, the strongest degradation appears under:

- weak motors
- low friction
- heavy mass

The policy is strongest under:

- stronger motors
- lighter body mass
- downhill slope

## Benchmark Role

This baseline is the fixed comparison point for:

- Baseline 3:
  - imitation-regularized blind baseline
- privileged teacher:
  - upper-bound policy with privileged information
- student:
  - deployable adaptation policy

Interpretation rule:

- improvements over this baseline should be discussed in terms of tracking,
  failure robustness, mismatch generalization, control effort, and behavior
  quality
- not reward alone

## Freeze Statement

Baseline 2 is now frozen.

Do not change:

- its checkpoint identity
- its warm-start source
- its training configuration
- its evaluation suite

If a stronger warm-start blind baseline is trained later, it must be created as
an explicitly new baseline version rather than silently replacing this one.
