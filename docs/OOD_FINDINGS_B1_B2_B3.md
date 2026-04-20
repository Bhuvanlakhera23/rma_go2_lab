# OOD Findings: Baseline 1 vs Baseline 2 vs Baseline 3

This note summarizes the first exploratory out-of-distribution probe across the
full frozen blind baseline ladder:

- Baseline 1: scratch blind rough baseline
- Baseline 2: warm-start blind rough baseline
- Baseline 3: warm-start + temporary imitation blind rough baseline

This document is exploratory.

It is intentionally separate from:

- `docs/BASELINE_COMPARISON_FINAL.md`
- `rma_go2_lab/policies/blind_baseline_protocol.md`
- `artifacts/evaluations/`

Use it to understand failure boundaries and OOD behavior, not to rewrite the
canonical baseline ranking silently.

## Scope

Compared checkpoints:

- Baseline 1:
  - `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- Baseline 2:
  - `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- Baseline 3:
  - `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

Exploratory suites:

- `ood_geometry_v1`
- `ood_dynamics_v1`
- `ood_push_v1`
- `ood_switch_v1`

Primary artifact roots:

- `artifacts/ood_evaluations/baseline1/`
- `artifacts/ood_evaluations/baseline2/`
- `artifacts/ood_evaluations/baseline3/`
- `artifacts/ood_evaluations/clips/baseline1/`
- `artifacts/ood_evaluations/clips/baseline2/`
- `artifacts/ood_evaluations/clips/baseline3/`

## Reading Rule

For OOD results, velocity error alone is not enough.

Small tracking errors can be misleading if the policy is failing quickly.

So the most important readout is the combination of:

- `timeout_fraction_of_terminals`
- `base_contact_fraction_of_terminals`
- failure event count
- visual behavior from OOD clips

Higher timeout fraction is better.

Higher base-contact fraction is worse.

## Geometry OOD Summary

Files:

- `artifacts/ood_evaluations/baseline1/ood_suite_blind_baseline1_scratch_final_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline2/ood_suite_blind_baseline2_warmstart_final_ood_geometry_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline3/ood_suite_blind_baseline3_warmstart_imitation_final_ood_geometry_v1_normal_seed999.json`

### Stairs Down

- Baseline 1:
  - `vel_err = 0.1372`
  - `timeout_frac = 0.5529`
  - `base_contact_frac = 0.4471`
- Baseline 2:
  - `vel_err = 0.2028`
  - `timeout_frac = 0.7222`
  - `base_contact_frac = 0.2778`
- Baseline 3:
  - `vel_err = 0.0896`
  - `timeout_frac = 0.7714`
  - `base_contact_frac = 0.2286`

Interpretation:

- all three handle stairs-down better than stairs-up
- Baseline 3 is best
- Baseline 2 is second
- Baseline 1 lags clearly on survival quality

### Random Rough Level 9

- Baseline 1:
  - `vel_err = 0.1726`
  - `timeout_frac = 0.3137`
  - `base_contact_frac = 0.6863`
- Baseline 2:
  - `vel_err = 0.1383`
  - `timeout_frac = 0.2241`
  - `base_contact_frac = 0.7759`
- Baseline 3:
  - `vel_err = 0.1674`
  - `timeout_frac = 0.2500`
  - `base_contact_frac = 0.7500`

Interpretation:

- this near-OOD harsher roughness case is mixed
- Baseline 1 actually survives relatively well here
- Baseline 2 tracks best
- Baseline 3 sits between them

### Boxes

- Baseline 1:
  - `vel_err = 0.1263`
  - `timeout_frac = 0.1810`
  - `base_contact_frac = 0.8190`
- Baseline 2:
  - `vel_err = 0.1342`
  - `timeout_frac = 0.1395`
  - `base_contact_frac = 0.8605`
- Baseline 3:
  - `vel_err = 0.1323`
  - `timeout_frac = 0.1849`
  - `base_contact_frac = 0.8151`

Interpretation:

- boxes are hard for all three
- Baseline 3 is best overall
- Baseline 1 is surprisingly competitive
- Baseline 2 is weakest of the three on this case

### Stairs Up

- Baseline 1:
  - `vel_err = 0.0794`
  - `timeout_frac = 0.0368`
  - `base_contact_frac = 0.9632`
- Baseline 2:
  - `vel_err = 0.0902`
  - `timeout_frac = 0.0585`
  - `base_contact_frac = 0.9415`
- Baseline 3:
  - `vel_err = 0.0757`
  - `timeout_frac = 0.0460`
  - `base_contact_frac = 0.9540`

Interpretation:

- stairs-up is harsh for all three
- all are overwhelmingly base-contact dominated
- Baseline 2 is slightly better on survival mix

### Geometry OOD Takeaway

Geometry OOD does not simply follow the canonical baseline ranking.

- Baseline 3 is the most geometry-interesting controller overall
- Baseline 2 is still better on stairs-up
- Baseline 1 is weaker overall, but not uniformly last

The strongest geometry-OOD advantages are:

- Baseline 3 on stairs-down and boxes
- Baseline 2 on stairs-up

## Dynamics OOD Summary

Files:

- `artifacts/ood_evaluations/baseline1/ood_suite_blind_baseline1_scratch_final_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline2/ood_suite_blind_baseline2_warmstart_final_ood_dynamics_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline3/ood_suite_blind_baseline3_warmstart_imitation_final_ood_dynamics_v1_normal_seed999.json`

### Ultra-High Friction

- Baseline 1:
  - `vel_err = 0.1934`
  - `timeout_frac = 0.2959`
  - `base_contact_frac = 0.7041`
- Baseline 2:
  - `vel_err = 0.1395`
  - `timeout_frac = 0.1618`
  - `base_contact_frac = 0.8382`
- Baseline 3:
  - `vel_err = 0.1419`
  - `timeout_frac = 0.1667`
  - `base_contact_frac = 0.8333`

Interpretation:

- Baseline 1 is best here
- Baseline 2 and Baseline 3 are effectively tied behind it

### Very Heavy

- Baseline 1:
  - `vel_err = 0.1003`
  - `timeout_frac = 0.1885`
  - `base_contact_frac = 0.8115`
- Baseline 2:
  - `vel_err = 0.1008`
  - `timeout_frac = 0.0314`
  - `base_contact_frac = 0.9686`
- Baseline 3:
  - `vel_err = 0.0850`
  - `timeout_frac = 0.0492`
  - `base_contact_frac = 0.9508`

Interpretation:

- Baseline 1 is best on heavy-mass OOD
- Baseline 3 is second
- Baseline 2 is weakest here

### Ultra-Low Friction

- Baseline 1:
  - `vel_err = 0.0319`
  - `timeout_frac = 0.0286`
  - `base_contact_frac = 0.9714`
- Baseline 2:
  - `vel_err = 0.0475`
  - `timeout_frac = 0.0486`
  - `base_contact_frac = 0.9514`
- Baseline 3:
  - `vel_err = 0.0370`
  - `timeout_frac = 0.0314`
  - `base_contact_frac = 0.9686`

Interpretation:

- all are highly slip-sensitive here
- Baseline 2 is best on survival mix
- Baseline 1 and Baseline 3 are worse

### Very Weak Motors

- Baseline 1:
  - `terminal_dones = 1581`
  - `timeout_frac = 0.0032`
  - `base_contact_frac = 0.9968`
  - `base_contact_events_per_env = 24.63`
- Baseline 2:
  - `terminal_dones = 1260`
  - `timeout_frac = 0.0063`
  - `base_contact_frac = 0.9937`
  - `base_contact_events_per_env = 19.56`
- Baseline 3:
  - `terminal_dones = 1815`
  - `timeout_frac = 0.0033`
  - `base_contact_frac = 0.9967`
  - `base_contact_events_per_env = 28.27`

Interpretation:

- this is the clearest separation in the dynamics OOD set
- Baseline 2 is clearly best
- Baseline 1 is second
- Baseline 3 is worst

### Dynamics OOD Takeaway

Dynamics OOD is where the baseline identities diverge the most.

- Baseline 1 is surprisingly sturdy under high friction and heavy mass
- Baseline 2 is the safest all-around dynamics baseline
- Baseline 3 is most vulnerable under severe actuator weakness

## Push Recovery OOD Summary

Files:

- `artifacts/ood_evaluations/baseline1/ood_suite_blind_baseline1_scratch_final_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline2/ood_suite_blind_baseline2_warmstart_final_ood_push_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline3/ood_suite_blind_baseline3_warmstart_imitation_final_ood_push_v1_normal_seed999.json`

### Yaw Push Medium

- Baseline 1:
  - `vel_err = 0.1557`
  - `timeout_frac = 0.2323`
  - `base_contact_frac = 0.7677`
- Baseline 2:
  - `vel_err = 0.1441`
  - `timeout_frac = 0.1901`
  - `base_contact_frac = 0.8099`
- Baseline 3:
  - `vel_err = 0.1561`
  - `timeout_frac = 0.2936`
  - `base_contact_frac = 0.7064`

Interpretation:

- yaw push is the easiest push case for all three
- Baseline 3 is best on recovery mix

### Forward Push Medium

- Baseline 1:
  - `vel_err = 0.1906`
  - `timeout_frac = 0.1833`
  - `base_contact_frac = 0.8167`
- Baseline 2:
  - `vel_err = 0.1593`
  - `timeout_frac = 0.0530`
  - `base_contact_frac = 0.9470`
- Baseline 3:
  - `vel_err = 0.2114`
  - `timeout_frac = 0.2393`
  - `base_contact_frac = 0.7607`

Interpretation:

- forward shove is one of the harshest push cases
- Baseline 3 is clearly best on survival mix
- Baseline 2 is weakest here

### Lateral Push Medium

- Baseline 1:
  - `vel_err = 0.1843`
  - `timeout_frac = 0.1913`
  - `base_contact_frac = 0.8087`
- Baseline 2:
  - `vel_err = 0.1635`
  - `timeout_frac = 0.0909`
  - `base_contact_frac = 0.9091`
- Baseline 3:
  - `vel_err = 0.1567`
  - `timeout_frac = 0.1259`
  - `base_contact_frac = 0.8741`

Interpretation:

- Baseline 1 is best here
- Baseline 3 is second
- Baseline 2 is weakest

### Lateral Push Repeated

- Baseline 1:
  - `vel_err = 0.1454`
  - `timeout_frac = 0.1638`
  - `base_contact_frac = 0.8362`
- Baseline 2:
  - `vel_err = 0.1641`
  - `timeout_frac = 0.1203`
  - `base_contact_frac = 0.8797`
- Baseline 3:
  - `vel_err = 0.2114`
  - `timeout_frac = 0.1652`
  - `base_contact_frac = 0.8348`

Interpretation:

- Baseline 1 and Baseline 3 are effectively tied on survival mix
- Baseline 2 trails both

### Push Recovery Takeaway

Push recovery does not follow the same ordering as static dynamics OOD.

- Baseline 3 is the strongest recovery-oriented baseline in this first push probe
- Baseline 1 is competitive and sometimes best under lateral pushes
- Baseline 2, despite being the safest all-around static-mismatch baseline, is weakest under these push disturbances

## Switch OOD Summary

Files:

- `artifacts/ood_evaluations/baseline1/ood_suite_blind_baseline1_scratch_final_ood_switch_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline2/ood_suite_blind_baseline2_warmstart_final_ood_switch_v1_normal_seed999.json`
- `artifacts/ood_evaluations/baseline3/ood_suite_blind_baseline3_warmstart_imitation_final_ood_switch_v1_normal_seed999.json`

For switch probes, read the `post_switch_metrics` first. That is the actual
recovery window after the regime change.

### Switch To Ultra-Low Friction

- Baseline 1:
  - `post_timeout_frac = 0.4844`
  - `post_contact_frac = 0.5156`
  - `post_vel_err = 0.2323`
- Baseline 2:
  - `post_timeout_frac = 0.3235`
  - `post_contact_frac = 0.6765`
  - `post_vel_err = 0.1612`
- Baseline 3:
  - `post_timeout_frac = 0.3659`
  - `post_contact_frac = 0.6341`
  - `post_vel_err = 0.1593`

Interpretation:

- this is the cleanest recovery-oriented switch case
- Baseline 1 is best on survival mix
- Baseline 3 is second
- Baseline 2 is third

### Switch To Low-Friction + Heavy

- Baseline 1:
  - `post_timeout_frac = 0.2088`
  - `post_contact_frac = 0.7912`
- Baseline 2:
  - `post_timeout_frac = 0.1204`
  - `post_contact_frac = 0.8796`
- Baseline 3:
  - `post_timeout_frac = 0.1172`
  - `post_contact_frac = 0.8828`

Interpretation:

- combined mid-episode switch is hard for all three
- Baseline 1 remains clearly best
- Baseline 2 and Baseline 3 are close behind

### Switch To Very Heavy

- Baseline 1:
  - `post_timeout_frac = 0.2524`
  - `post_contact_frac = 0.7476`
- Baseline 2:
  - `post_timeout_frac = 0.1241`
  - `post_contact_frac = 0.8759`
- Baseline 3:
  - `post_timeout_frac = 0.0828`
  - `post_contact_frac = 0.9172`

Interpretation:

- this is the strongest separation in the switch set
- Baseline 1 is best
- Baseline 2 is second
- Baseline 3 is weakest

### Switch To Very Weak Motors

- Baseline 1:
  - `post_timeout_frac = 0.0058`
  - `post_contact_frac = 0.9942`
  - `post_terminal_dones = 1043`
- Baseline 2:
  - `post_timeout_frac = 0.0089`
  - `post_contact_frac = 0.9911`
  - `post_terminal_dones = 790`
- Baseline 3:
  - `post_timeout_frac = 0.0050`
  - `post_contact_frac = 0.9950`
  - `post_terminal_dones = 1200`

Interpretation:

- all three are highly vulnerable to abrupt weak-motor switches
- Baseline 2 is best here
- Baseline 3 is worst
- tiny tracking errors are not meaningful in this catastrophic case

### Switch OOD Takeaway

Switch OOD produces yet another ordering.

- Baseline 1 is the strongest abrupt regime-switch baseline overall
- Baseline 2 is second overall and best only on weak-motor switch
- Baseline 3, despite strong push recovery, is weakest under mid-episode switch shocks

## Visual Clip Guidance

The recorded OOD clips are for behavior inspection, not canonical scoring.

These clips were recorded with:

- fixed forward command
- selected termination terms disabled for visualization only

That makes them useful for seeing traversal and failure style, but not for
replacing the suite metrics.

Useful clip sets:

- Baseline 2:
  - `artifacts/ood_evaluations/clips/baseline2/`
- Baseline 3:
  - `artifacts/ood_evaluations/clips/baseline3/`
- Baseline 1:
  - `artifacts/ood_evaluations/clips/baseline1/`

Recommended first clips to inspect:

- stairs up
- boxes
- ultra-low friction
- very weak motors
- very heavy
- switch to ultra-low friction

## Overall Interpretation

The OOD picture is more nuanced than the in-distribution baseline story.

- Baseline 2 remains the safest all-around controller.
- Baseline 3 is the strongest geometry-interesting variant.
- Baseline 1 is weaker in the canonical benchmark, but not uniformly weakest
  under OOD dynamics.
- Baseline 3 is currently the strongest push-recovery baseline in `ood_push_v1`.
- Baseline 1 is currently the strongest abrupt-switch baseline in `ood_switch_v1`.

So the current best interpretation is:

- Baseline 2:
  - best dependable default baseline
- Baseline 3:
  - best geometry-shift exploratory variant
  - best first-pass push-recovery variant
- Baseline 1:
  - lower canonical performer, but with some unexpectedly sturdy dynamics OOD
    behavior
  - best current switch-recovery baseline

## What This Does Not Mean

This does not change the canonical baseline ranking automatically.

It also does not prove that one baseline is globally best under every OOD
regime.

It means:

- geometry OOD and dynamics OOD pull in different directions
- push recovery adds a third axis that again changes the ordering
- switch recovery adds a fourth axis that changes the ordering again
- the current frozen ladder has different failure personalities
- future teacher or perception branches should remember that actuator weakness
  remains a major blind-policy vulnerability

## Recommended Next Step

If OOD probing continues, the next useful addition is:

- terrain transitions

Until then, the main actionable takeaway is:

- keep Baseline 2 as the safer default reference
- keep Baseline 3 as the more geometry-interesting variant
- remember that Baseline 1 is still scientifically useful as a contrasting OOD
  failure style
