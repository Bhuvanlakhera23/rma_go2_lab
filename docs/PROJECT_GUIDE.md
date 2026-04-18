# RMA-Go2 Project Guide

This is the shortest reliable way to understand the repo.

Use this file first. Then jump to the linked detailed docs only when needed.

## Project Goal

The active project goal is to build and compare a clean blind locomotion ladder:

1. flat prior
2. blind rough scratch baseline
3. blind rough warm-start baseline
4. blind rough warm-start + imitation baseline

The core scientific question for the active repo state is:

> how much do warm-starting and flat-prior imitation help a fixed blind rough
> locomotion controller under a frozen benchmark?

## Canonical Experiment Ladder

### Flat prior

- Task: `RMA-Go2-Flat`
- Purpose: train a clean locomotion prior used to initialize later runs
- Status: validated checkpoint already selected

Selected flat prior:

- `rma_go2_lab/policies/flat1499.pt`
- sanity reports:
  - `artifacts/evaluations/flat_prior/gait_flat_prior_model1500_standstill.json`
  - `artifacts/evaluations/flat_prior/gait_flat_prior_model1500_forward.json`

### Blind baselines

1. `RMA-Go2-Blind-Baseline-Rough`
   - rough blind scratch baseline
2. `RMA-Go2-Blind-Baseline-Rough-WarmStart`
   - same rough blind baseline, actor warm-started from the flat prior
3. `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation`
   - same rough blind baseline, warm-start plus temporary imitation prior

Blind baselines are comparison baselines, not moving targets.

Current frozen Baseline 1 checkpoint:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`

Current frozen Baseline 2 checkpoint:

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

## Governing Principle For Blind Baselines

Blind baselines should not be trained to become unbeatable obstacle
specialists.

They should be trained as competent fixed controllers, then evaluated under
controlled hidden mismatch:

- friction
- mass
- motor strength
- terrain geometry
- mid-episode changes

Canonical SOP:

- `rma_go2_lab/policies/blind_baseline_protocol.md`

## Where Things Live

### Environments

- `rma_go2_lab/envs/priors/`
  - shared flat-prior envs
- `rma_go2_lab/envs/blind/`
  - blind baseline envs

### Models

- `rma_go2_lab/models/priors/`
  - shared flat-prior PPO configs
- `rma_go2_lab/models/blind/`
  - blind PPO configs, warm-start actor-critic, imitation PPO

Teacher/student legacy code is not part of the active branch anymore.

### Evaluators

- `scripts/eval/gait.py`
  - gait, standstill, step response, forward drift checks
- `scripts/eval/isolated.py`
  - single controlled evaluation scenario
- `scripts/eval/run_isolated_suite.py`
  - suite runner across many controlled scenarios
- `scripts/eval/blind_baseline_diagnostics.py`
  - blind baseline health diagnostics

### Export / evaluation artifacts

- `scripts/export/`
- `artifacts/evaluations/`

Note:

- `logs/` at the repo root is a local symlink to the IsaacLab run directory.
  It is generated convenience state, not source code.

## Active Files That Matter Most

If you only open a few files, open these:

- `rma_go2_lab/__init__.py`
  - active registered tasks only
- `rma_go2_lab/envs/priors/flat_cfg.py`
  - current flat-prior environment
- `rma_go2_lab/models/priors/flat_ppo_cfg.py`
  - current flat-prior PPO recipe
- `rma_go2_lab/envs/blind/rough_cfg.py`
  - current shared rough environment for Baseline 1, Baseline 2, and Baseline 3
- `rma_go2_lab/models/blind/variants_ppo_cfg.py`
  - blind PPO ladder and warm-start checkpoint wiring
- `rma_go2_lab/models/blind/ppo_with_flat_expert.py`
  - imitation variant

## How To Read A Training Run

Look at these first:

- `track_lin_vel_xy_exp`
- `Metrics/base_velocity/error_vel_xy`
- `Episode_Termination/time_out`
- `Episode_Termination/base_height`
- `Episode_Termination/base_orientation`
- `Episode_Termination/low_progress`
- `Curriculum/terrain_levels`

Interpretation:

- high tracking + high timeout + low failure terms:
  healthy run
- low progress high:
  stuck policy
- base height high:
  terrain-clearance / body-clearance problem
- base orientation high:
  tipping / posture instability problem

## How To Judge A Baseline

Do not judge a baseline mainly by reward.

Use:

- tracking error
- time-to-failure
- failure-cause distribution
- recovery under mismatch
- slip / drift
- action effort

Reward is for training. Degradation under mismatch is the research result.

## Recommended Reading Order

1. `docs/PROJECT_GUIDE.md`
2. `rma_go2_lab/policies/blind_baseline_protocol.md`
3. `rma_go2_lab/policies/README.md`
4. `artifacts/evaluations/README.md`
5. task-specific env / PPO config files

## What Not To Do

- do not keep reinventing blind-baseline reward design every few days
- do not merge every reward idea from every reference repo
- do not broaden command distributions casually
- do not use raw reward as proof that RMA is unnecessary

## One-Line Mental Model

This repo is organized around one clean story:

> establish fixed blind baselines, measure their degradation under hidden
> mismatch, then justify and test adaptation on the same tasks.
