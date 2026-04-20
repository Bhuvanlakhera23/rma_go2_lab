# OOD Probe Protocol

This document defines the non-canonical exploratory evaluation path for the
frozen proprio baseline ladder.

Use this path when the goal is:

- to probe the limits of frozen baselines
- to test out-of-distribution terrain or dynamics
- to understand failure boundaries
- to generate hypotheses for future branches

Do **not** use this path to silently change the canonical baseline comparison.

## Status

This OOD path is exploratory.

It is intentionally separate from:

- `rma_go2_lab/policies/blind_baseline_protocol.md`
- `docs/BASELINE_COMPARISON_FINAL.md`
- `artifacts/evaluations/`

Canonical baseline claims must continue to come from the frozen protocol and
the frozen artifact set, not from this exploratory path.

## Governing Rule

OOD probing is allowed to be aggressive, creative, and broad.

But every OOD result must be interpreted as:

- exploratory
- non-canonical
- not automatically publishable as baseline-comparison evidence

An OOD finding only moves into the main story if we explicitly promote it
later.

## Separate Storage

Use the following paths only for OOD work:

- `artifacts/ood_evaluations/`
- `scripts/eval_ood/`

Do not write OOD outputs into:

- `artifacts/evaluations/`

unless we deliberately decide to version and promote a probe into the canonical
pipeline.

## Recommended OOD Buckets

### Geometry OOD

- stairs
- inverse stairs
- boxes
- stepping-stone-like or discrete foothold terrain
- mixed terrain transitions
- roughness amplitude beyond training regime

### Dynamics OOD

- ultra-low friction
- ultra-high friction
- heavy mass beyond frozen suite
- weak motors beyond frozen suite
- combined friction + mass + motor perturbations

### Recovery OOD

- lateral push disturbances
- forward shove disturbances
- yaw / heading impulse disturbances
- repeated small pushes
- recovery after non-trained external disturbance

### Switch OOD

- sudden friction drop mid-episode
- sudden mass increase mid-episode
- sudden actuator weakening mid-episode
- combined regime switch during ongoing traversal

### Horizon OOD

- longer episodes than canonical evaluation
- repeated disturbance exposure
- curriculum levels or terrain severity beyond the frozen benchmark

## OOD Baseline Set

Unless there is a special reason not to, the default OOD comparison set should
be:

- Baseline 1
- Baseline 2
- Baseline 3

The flat prior may be included for perspective, but it is not part of the main
blind baseline ladder.

## OOD Metric Policy

Use the same core metric philosophy as the canonical protocol:

- tracking quality
- time-to-failure
- failure-cause distribution
- drift / slip
- body stability
- control effort proxies

Additional OOD-specific metrics are allowed, for example:

- terrain-family success rate
- stair-step completion fraction
- transition survival rate
- longest uninterrupted traversal
- distance covered before first catastrophic failure

If a new metric is introduced for OOD, document:

- what it measures
- why it matters
- whether larger or smaller is better

## OOD Naming Rules

Every OOD artifact name should encode:

- baseline identity
- scenario identity
- whether it is canonical or exploratory

Examples:

- `ood_stairs_level5_baseline2.json`
- `ood_boxes_sparse_baseline3.csv`
- `ood_transition_flat_to_stairs_baseline1.mp4`

Avoid ambiguous names like:

- `weird_test.json`
- `final_stairs.csv`

## Promotion Rule

An OOD result should only be promoted into the main repo story if:

1. it answers a real scientific question we now care about
2. it is reproducible
3. it is run across the frozen baseline set, not cherry-picked
4. it is documented as a new explicit protocol version

Until then, treat OOD probing as:

- valuable
- informative
- but non-canonical

## Current Recommendation

Use the OOD path for:

- stairs
- boxes
- harsher random rough
- mixed geometry transitions
- extreme dynamics combinations
- push disturbance recovery
- mid-episode dynamics switches

Keep the main baseline ladder exactly as frozen.

## Current Findings

First comparative findings for the frozen warm-start baselines are summarized
in:

- `docs/OOD_FINDINGS_B1_B2_B3.md`

That note should be treated as:

- exploratory
- non-canonical
- useful for planning later branches, especially teacher and perception work
