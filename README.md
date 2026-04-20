# RMA-Go2 Lab

Frozen proprioceptive quadruped locomotion baselines for rough terrain.

This repo asks one focused question:

> How much do warm-starting and temporary imitation help a fixed blind rough-terrain controller under a frozen benchmark?

The branch is intentionally **baseline-first**. It is not a moving-target playground and it is not the old teacher/student pipeline resurrected in disguise. The point is to build a clean reference ladder, evaluate it honestly, and carry those conclusions forward into the next phase.

## What This Repo Contains

The active frozen ladder is:

1. **Flat prior**
   - nominal locomotion prior on flat terrain
2. **Baseline 1**
   - scratch blind rough controller
3. **Baseline 2**
   - warm-started from the flat prior
4. **Baseline 3**
   - warm-started plus temporary imitation from the flat prior

Frozen checkpoints:

- `rma_go2_lab/policies/flat1499.pt`
- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

## Main Results

### Canonical in-distribution story

| Baseline | Role | Main takeaway |
| --- | --- | --- |
| **B1** | Scratch reference | Honest lower bound and useful OOD contrast case |
| **B2** | Warm-start baseline | Cleanest and safest canonical blind baseline |
| **B3** | Warm-start + imitation | Stronger robustness-leaning variant, but less clean than B2 |

### OOD winner by axis

| Evaluation axis | Best baseline | Interpretation |
| --- | --- | --- |
| Geometry OOD | **B3** | Strongest on stairs-down and boxes |
| Static dynamics OOD | **B2** | Safest all-around under fixed hidden mismatch |
| Push recovery OOD | **B3** | Best disturbance recovery in first push probe |
| Mid-episode switch OOD | **B1** | Best abrupt regime-switch resilience |

### Practical read

- **B2** is the canonical proprio-only reference.
- **B3** is the geometry/push-recovery variant.
- **B1** is the switch-resilient scratch reference.

This repo does **not** claim that one baseline wins every axis. The more useful result is that different interventions improve different robustness families.

## Watch The Behavior

### Canonical videos

- Flat prior nominal hero:
  - [`artifacts/evaluations/clips/flat_prior/nominal_hero_20260418_122357/nominal_hero_20260418_122357.mp4`](artifacts/evaluations/clips/flat_prior/nominal_hero_20260418_122357/nominal_hero_20260418_122357.mp4)
- Baseline 1 nominal hero:
  - [`artifacts/evaluations/clips/baseline1/nominal_hero_20260418_100756/nominal_hero_20260418_100756.mp4`](artifacts/evaluations/clips/baseline1/nominal_hero_20260418_100756/nominal_hero_20260418_100756.mp4)
- Baseline 2 nominal hero:
  - [`artifacts/evaluations/clips/baseline2/nominal_hero_20260417_173511/nominal_hero_20260417_173511.mp4`](artifacts/evaluations/clips/baseline2/nominal_hero_20260417_173511/nominal_hero_20260417_173511.mp4)
- Baseline 3 nominal hero:
  - [`artifacts/evaluations/clips/baseline3/model560_nominal_hero_20260420_102922/model560_nominal_hero_20260420_102922.mp4`](artifacts/evaluations/clips/baseline3/model560_nominal_hero_20260420_102922/model560_nominal_hero_20260420_102922.mp4)

### OOD videos

- B3 stairs-down geometry OOD:
  - [`artifacts/ood_evaluations/clips/baseline3/ood_stairs_down_hero_20260420_130307/ood_stairs_down_hero_20260420_130307.mp4`](artifacts/ood_evaluations/clips/baseline3/ood_stairs_down_hero_20260420_130307/ood_stairs_down_hero_20260420_130307.mp4)
- B2 ultra-low-friction dynamics OOD:
  - [`artifacts/ood_evaluations/clips/baseline2/ood_ultra_low_friction_hero_20260420_122200/ood_ultra_low_friction_hero_20260420_122200.mp4`](artifacts/ood_evaluations/clips/baseline2/ood_ultra_low_friction_hero_20260420_122200/ood_ultra_low_friction_hero_20260420_122200.mp4)
- B3 yaw-push recovery OOD:
  - [`artifacts/ood_evaluations/clips/baseline3/ood_push_yaw_medium_hero_20260420_135550/ood_push_yaw_medium_hero_20260420_135550.mp4`](artifacts/ood_evaluations/clips/baseline3/ood_push_yaw_medium_hero_20260420_135550/ood_push_yaw_medium_hero_20260420_135550.mp4)
- B1 ultra-low-friction switch OOD:
  - [`artifacts/ood_evaluations/clips/baseline1/ood_switch_ultra_low_friction_hero_20260420_142815/ood_switch_ultra_low_friction_hero_20260420_142815.mp4`](artifacts/ood_evaluations/clips/baseline1/ood_switch_ultra_low_friction_hero_20260420_142815/ood_switch_ultra_low_friction_hero_20260420_142815.mp4)

## Repo Reading Path

If you land here fresh, read in this order:

1. [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)
2. [`rma_go2_lab/policies/blind_baseline_protocol.md`](rma_go2_lab/policies/blind_baseline_protocol.md)
3. [`docs/BASELINE_COMPARISON_FINAL.md`](docs/BASELINE_COMPARISON_FINAL.md)
4. [`docs/OOD_FINDINGS_B1_B2_B3.md`](docs/OOD_FINDINGS_B1_B2_B3.md)
5. [`docs/FROZEN_BASELINE_SYNTHESIS.md`](docs/FROZEN_BASELINE_SYNTHESIS.md)
6. [`docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md`](docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md)
7. [`docs/BASELINE_REGIME_CLOSED.md`](docs/BASELINE_REGIME_CLOSED.md)

## Evaluation Philosophy

This repo does **not** treat reward as the final result.

Baselines are judged by:

- tracking quality
- failure behavior
- drift and slip
- gait structure
- static mismatch tolerance
- push recovery
- abrupt regime-switch resilience

That is why the baseline story is split into:

- **canonical frozen comparison** under the main benchmark
- **exploratory OOD probing** under separate protocols

## Project Layout

### Main code paths

- [`rma_go2_lab/envs/priors/flat_cfg.py`](rma_go2_lab/envs/priors/flat_cfg.py)
  - flat-prior environment
- [`rma_go2_lab/models/priors/flat_ppo_cfg.py`](rma_go2_lab/models/priors/flat_ppo_cfg.py)
  - flat-prior PPO recipe
- [`rma_go2_lab/envs/blind/rough_cfg.py`](rma_go2_lab/envs/blind/rough_cfg.py)
  - shared rough environment for B1, B2, and B3
- [`rma_go2_lab/models/blind/variants_ppo_cfg.py`](rma_go2_lab/models/blind/variants_ppo_cfg.py)
  - baseline ladder configs
- [`rma_go2_lab/models/blind/ppo_with_flat_expert.py`](rma_go2_lab/models/blind/ppo_with_flat_expert.py)
  - imitation-augmented PPO for B3

### Artifact roots

- [`artifacts/evaluations/`](artifacts/evaluations/)
  - canonical benchmark artifacts
- [`artifacts/ood_evaluations/`](artifacts/ood_evaluations/)
  - exploratory OOD artifacts
- [`rma_go2_lab/policies/`](rma_go2_lab/policies/)
  - frozen checkpoints and freeze notes

## Documentation Map

### Canonical baseline docs

- [`docs/BASELINE_COMPARISON_FINAL.md`](docs/BASELINE_COMPARISON_FINAL.md)
- [`docs/FROZEN_BASELINE_SYNTHESIS.md`](docs/FROZEN_BASELINE_SYNTHESIS.md)
- [`docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md`](docs/FROZEN_BASELINE_RESULTS_AT_A_GLANCE.md)
- [`docs/BASELINE_REGIME_CLOSED.md`](docs/BASELINE_REGIME_CLOSED.md)

### OOD docs

- [`docs/OOD_PROBE_PROTOCOL.md`](docs/OOD_PROBE_PROTOCOL.md)
- [`docs/OOD_FINDINGS_B1_B2_B3.md`](docs/OOD_FINDINGS_B1_B2_B3.md)

### Artifact indexes

- [`artifacts/evaluations/README.md`](artifacts/evaluations/README.md)
- [`artifacts/ood_evaluations/README.md`](artifacts/ood_evaluations/README.md)

## Quick Start

Train the flat prior:

```bash
./isaaclab.sh -p /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task RMA-Go2-Flat
```

Train Baseline 1:

```bash
./isaaclab.sh -p /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task RMA-Go2-Blind-Baseline-Rough
```

Train Baseline 2:

```bash
./isaaclab.sh -p /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart
```

Train Baseline 3:

```bash
./isaaclab.sh -p /home/bhuvan/tools/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation
```

Run the canonical mismatch suite:

```bash
./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval/run_isolated_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --suite blind_baseline_v1
```

Run an exploratory OOD suite:

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt --suite ood_geometry_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/baseline2
```

## Status

The baseline regime is closed and frozen.

That means:

- the baseline ladder is no longer a moving target
- the repo now has a stable reference frame for later teacher, adaptation, or perception work

If you want the shortest project-management summary, start here:

- [`docs/BASELINE_REGIME_CLOSED.md`](docs/BASELINE_REGIME_CLOSED.md)
