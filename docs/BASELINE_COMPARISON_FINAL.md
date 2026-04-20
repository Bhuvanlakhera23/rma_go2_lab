# Baseline Comparison Final

This document is the research-facing comparison note for the frozen proprio
baseline ladder.

It turns the protocol principles in
`rma_go2_lab/policies/blind_baseline_protocol.md` into explicit numbers for the
three frozen baselines:

- Baseline 1: scratch
- Baseline 2: warm-start
- Baseline 3: warm-start + temporary imitation

## Frozen Checkpoints

- Baseline 1:
  - `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
  - selected from `model_1999.pt`
- Baseline 2:
  - `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
  - selected from `model_1500.pt`
- Baseline 3:
  - `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`
  - selected from `model_560.pt` after a checkpoint sweep

## Metric Mapping

The blind-baseline protocol defined the primary comparison categories as:

- velocity tracking error
- time-to-failure
- failure-cause distribution
- slip and drift
- body-height and orientation instability
- action effort / energy proxy

For the frozen baseline comparison, these categories are instantiated as:

| Comparison category | Operational metric(s) used here | Interpretation |
| --- | --- | --- |
| Tracking quality | `vel_err_step_mean`, `planar_speed_mean`, `command_planar_speed_mean` | Lower velocity error and higher achieved speed at the same command are better |
| Survival / time-to-failure | `timeout_fraction_of_terminals` | Higher means more episodes survive to timeout instead of failing early |
| Failure mode | `base_contact_fraction_of_terminals` | Lower means fewer failures terminate by base contact |
| Drift / locomotion quality | `forward_lateral_drift_per_meter_mean`, `diagonal_trot_score`, `fore_hind_swing_exact_pair_fraction` | Lower drift, higher diagonal tendency, lower fore-hind pairing are better |
| Standstill quality | `standstill_planar_speed_mean`, `standstill_foot_speed_mean`, `standstill_joint_vel_abs_mean` | Lower is better |
| Body stability | `base_tilt_projected_gravity_xy_mean` | Lower is better |

Notes:

- Reward is not used as the main scientific metric.
- “Success rate” is represented here by survival-to-timeout behavior rather than
  a separate binary success label.
- “Distance covered” is represented indirectly through achieved planar speed at
  the frozen forward command.
- “Smoothness” is represented indirectly through standstill motion, drift, and
  body-stability proxies. A dedicated single smoothness scalar is not yet part
  of the frozen artifact set.

## Frozen Eval Summary

### Standstill quality

| Baseline | Standstill planar speed | Standstill foot speed | Standstill joint vel | Standstill base tilt |
| --- | ---: | ---: | ---: | ---: |
| B1 scratch | 0.0250 | 0.0594 | 0.2863 | 0.0389 |
| B2 warm-start | 0.0259 | 0.0558 | 0.2619 | 0.0466 |
| B3 imitation | 0.0284 | 0.0545 | 0.2701 | 0.0396 |

Read:

- B2 is the quietest on joint motion.
- B3 is slightly worse than B2 at standstill, but still better than B1 on foot
  speed.
- B1 is the noisiest scratch controller.

### Forward-motion quality

| Baseline | Achieved speed | Commanded speed | Diagonal trot score | Fore-hind exact pair | Drift per meter | Forward base tilt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 scratch | 0.3506 | 0.7497 | 0.0416 | 0.1997 | 2.5436 | 0.2539 |
| B2 warm-start | 0.3749 | 0.7490 | 0.1402 | 0.0350 | 1.6319 | 0.1573 |
| B3 imitation | 0.3677 | 0.7492 | 0.1691 | 0.0669 | 2.3628 | 0.1836 |

Read:

- B2 is the cleanest forward controller overall.
- B3 has the strongest diagonal tendency of the three, but still drifts much
  more than B2.
- B1 is the clearest fore-hind / bound-prone baseline.

### Suite-level mismatch summary

| Baseline | Avg vel err | Avg timeout frac | Avg base-contact frac |
| --- | ---: | ---: | ---: |
| B1 scratch | 0.1322 | 0.2566 | 0.7434 |
| B2 warm-start | 0.1502 | 0.2806 | 0.7194 |
| B3 imitation | 0.1368 | 0.2990 | 0.7010 |

Read:

- B3 has the best overall survival/contact balance in the frozen suite.
- B2 is second-best on suite robustness balance.
- B1 is the weakest on overall survival/contact balance.

## Key Scenario Comparison

### Nominal random rough level 5

| Baseline | Vel err | Timeout frac | Base-contact frac |
| --- | ---: | ---: | ---: |
| B1 scratch | 0.1483 | 0.3474 | 0.6526 |
| B2 warm-start | 0.1412 | 0.2404 | 0.7596 |
| B3 imitation | 0.1290 | 0.1971 | 0.8029 |

Read:

- B3 has the best nominal velocity error.
- B1 has the highest nominal timeout fraction.
- B2 and B3 trade nominal tracking against more base-contact-dominated failure.

### Low friction random rough level 5

| Baseline | Vel err | Timeout frac | Base-contact frac |
| --- | ---: | ---: | ---: |
| B1 scratch | 0.0488 | 0.0483 | 0.9517 |
| B2 warm-start | 0.0746 | 0.1083 | 0.8917 |
| B3 imitation | 0.0551 | 0.0441 | 0.9559 |

Read:

- B2 is clearly best under low friction by failure composition.
- B1 and B3 both remain heavily base-contact dominated here.

### Heavy mass random rough level 5

| Baseline | Vel err | Timeout frac | Base-contact frac |
| --- | ---: | ---: | ---: |
| B1 scratch | 0.1173 | 0.1500 | 0.8500 |
| B2 warm-start | 0.1168 | 0.1224 | 0.8776 |
| B3 imitation | 0.0987 | 0.0642 | 0.9358 |

Read:

- B3 tracks best under heavy mass.
- B1 remains the least base-contact dominated in this one scenario.
- B2 is intermediate.

### Weak motor random rough level 5

| Baseline | Vel err | Timeout frac | Base-contact frac |
| --- | ---: | ---: | ---: |
| B1 scratch | 0.0597 | 0.0222 | 0.9778 |
| B2 warm-start | 0.0863 | 0.0287 | 0.9713 |
| B3 imitation | 0.0488 | 0.0208 | 0.9792 |

Read:

- All three are weak here.
- B3 has the best velocity error but not the best failure composition.
- weak motors remain a universal stressor across the whole ladder

## What Improved

### B1 to B2

Warm-start improved the baseline most clearly in behavior quality:

- diagonal tendency improved:
  - `0.0416 -> 0.1402`
- fore-hind exact pairing dropped:
  - `0.1997 -> 0.0350`
- drift improved:
  - `2.5436 -> 1.6319`
- forward base tilt improved:
  - `0.2539 -> 0.1573`
- standstill joint motion improved:
  - `0.2863 -> 0.2619`

Interpretation:

- B2 is the clearest improvement over scratch.
- warm-start is the strongest intervention for clean forward locomotion.

### B2 to B3

Temporary imitation improved robustness balance more than it improved clean
motion:

- diagonal trot score improved:
  - `0.1402 -> 0.1691`
- average timeout fraction improved:
  - `0.2806 -> 0.2990`
- average base-contact fraction improved:
  - `0.7194 -> 0.7010`

But B3 degraded relative to B2 on motion cleanliness:

- drift worsened:
  - `1.6319 -> 2.3628`
- forward base tilt worsened:
  - `0.1573 -> 0.1836`
- fore-hind exact pairing increased:
  - `0.0350 -> 0.0669`

Interpretation:

- B3 is the strongest robustness-leaning variant.
- B2 remains the cleaner deployable baseline.

## Final Interpretation

The frozen baseline ladder supports the following claims:

- Baseline 1 is the honest scratch reference.
- Baseline 2 is the best clean warm-start baseline.
- Baseline 3 is the best robustness-leaning variant, selected from a
  checkpoint sweep rather than from the last iteration.

The main research takeaway is:

- warm-start gives the clearest improvement in behavior quality
- temporary imitation changes the controller tradeoff rather than simply
  dominating warm-start on every metric

That tradeoff should be described honestly:

- B2 is cleaner
- B3 is tougher on the frozen mismatch suite

## Publishing Guidance

For research-facing reporting, use:

- Baseline 2 as the clean canonical baseline
- Baseline 3 as the stronger robustness-leaning comparison variant
- the protocol in `rma_go2_lab/policies/blind_baseline_protocol.md` as the
  methodological ground truth

Do not claim:

- that B3 is strictly better than B2 on all axes
- that reward alone determines the better baseline
- that the current ladder covers stairs, boxes, or perception-enabled control

Those belong to later extension studies.
