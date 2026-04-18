# Flat Prior Freeze

This file freezes the selected flat locomotion prior for the current project
phase.

The flat prior is now frozen. Do not silently replace it in place. If a later
flat prior is trained, it should be versioned explicitly.

## Identity

- canonical name: `flat1499`
- checkpoint: `rma_go2_lab/policies/flat1499.pt`
- source run:
  - `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_flat/2026-04-17_14-40-21`
- selected source checkpoint:
  - `model_1499.pt`

## Purpose

The flat prior is the shared nominal locomotion prior used to initialize later
blind baselines.

It represents:

- non-privileged flat-ground locomotion
- a clean nominal gait prior rather than a robustness benchmark
- the warm-start source for Baseline 2 and Baseline 3

It is not itself a full mismatch benchmark target.

## Training Definition

- task:
  - `RMA-Go2-Flat`
- env config:
  - `rma_go2_lab/envs/priors/flat_cfg.py`
- PPO config:
  - `rma_go2_lab/models/priors/flat_ppo_cfg.py`

Frozen characteristics retained in this version:

- actor/critic observation normalization disabled
- forward-only flat command regime
- mild friction randomization only
- no explicit base-height shaping
- `feet_air_time = 0.3`

## Selection Rationale

`model_1499.pt` from the `2026-04-17_14-40-21` run was selected because:

- training stayed numerically stable to the end of the run
- standstill behavior remained quiet and stable
- forward locomotion showed the desired diagonal-pair trot-like gait
- it clearly beat the sandbox variant behaviorally even when the sandbox looked
  slightly cleaner on scalar metrics

This freeze also closes the normalization branch for the flat prior family:
keeping observation normalization enabled changed the emergent gait in an
undesirable way for this prior.

## Final Evaluation Artifacts

Controller-quality sanity checks:

- `artifacts/evaluations/flat_prior/gait_flat_prior_model1500_standstill.json`
- `artifacts/evaluations/flat_prior/gait_flat_prior_model1500_forward.json`

## Final Behavior Summary

Training-side final regime near freeze:

- tracking error:
  - `error_vel_xy = 0.0970`
  - `error_vel_yaw = 0.1432`
- terminations:
  - `time_out = 1.0000`
  - `base_contact = 0.0000`

Standstill sanity:

- standstill planar speed:
  - `0.0171`
- standstill foot speed:
  - `0.0300`
- standstill joint velocity abs mean:
  - `0.1772`
- standstill base tilt proxy:
  - `0.0198`

Forward sanity:

- achieved planar speed:
  - `0.6138`
- commanded planar speed:
  - `0.7500`
- diagonal trot score:
  - `0.4053`
- diagonal exact-pair swing fraction:
  - `0.3409`
- forward lateral drift per meter:
  - `2.3530`
- base tilt proxy:
  - `0.0391`
- gait interpretation:
  - `high_duty_diagonal_gait_staggered_touchdown`

## Pipeline Role

This prior is the frozen locomotion seed for:

- Baseline 2:
  - blind warm-start baseline
- Baseline 3:
  - blind warm-start + imitation baseline

The blind baseline pipeline should reference the canonical archived checkpoint:

- `rma_go2_lab/policies/flat1499.pt`

## Freeze Statement

The flat prior is now frozen.

Do not change:

- its checkpoint identity
- its training recipe
- its role as the warm-start prior
- its sanity evaluation pair

If a stronger flat prior is trained later, it must be created as a new,
explicitly versioned artifact rather than replacing this one silently.
