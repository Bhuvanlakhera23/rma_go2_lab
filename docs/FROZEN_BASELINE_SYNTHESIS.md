# Frozen Baseline Synthesis

This document is the consolidated research-facing synthesis of the frozen
proprio baseline ladder.

It combines:

- the canonical in-distribution baseline comparison
- the exploratory OOD probes
- the main interpretation that emerges when those are read together

It does **not** replace the underlying protocol or artifact notes. Instead, it
answers the higher-level question:

> what did each frozen baseline actually buy us?

Primary source notes:

- `docs/BASELINE_COMPARISON_FINAL.md`
- `docs/OOD_FINDINGS_B1_B2_B3.md`
- `rma_go2_lab/policies/blind_baseline_protocol.md`
- `docs/OOD_PROBE_PROTOCOL.md`

## Frozen Baselines

- Baseline 1:
  - scratch proprio-only rough controller
  - `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- Baseline 2:
  - warm-started from the flat prior
  - `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- Baseline 3:
  - warm-started plus temporary flat-prior imitation
  - `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

## The Clean Canonical Story

Under the frozen in-distribution protocol, the baseline ranking is:

1. Baseline 2 as the cleanest deployable blind baseline
2. Baseline 3 as the stronger robustness-leaning variant
3. Baseline 1 as the honest scratch reference

Why Baseline 2 is the canonical winner:

- cleanest forward locomotion
- lowest drift
- best body attitude during forward motion
- strongest all-around behavior quality under the canonical benchmark

Why Baseline 3 is not the canonical winner:

- it improved suite-level robustness balance
- it improved diagonal tendency
- but it degraded motion cleanliness relative to Baseline 2

Why Baseline 1 matters:

- it anchors the scratch floor honestly
- it proves the warm-start gains are real rather than assumed

## What Each Intervention Changed

### Scratch to warm-start

The move from Baseline 1 to Baseline 2 gave the clearest single improvement in
the whole ladder.

Warm-start improved:

- forward gait organization
- drift control
- body stability in motion
- standstill quietness

This is the strongest positive result of the canonical ladder.

### Warm-start to warm-start plus imitation

The move from Baseline 2 to Baseline 3 did not simply produce a stronger
version of Baseline 2.

Instead, it changed the controller tradeoff:

- better robustness balance in the frozen mismatch suite
- stronger diagonal tendency
- messier locomotion style
- more drift than Baseline 2

So temporary imitation should be understood as a tradeoff-shaping intervention,
not a universal improvement.

## OOD Axes And Their Winners

The OOD probes show that there is no single universal winner across all stress
types.

### Geometry OOD

Main result:

- Baseline 3 is the strongest geometry-interesting variant overall

Notable cases:

- Baseline 3 is best on stairs-down and boxes
- Baseline 2 is slightly better on stairs-up
- Baseline 1 is not uniformly last

Interpretation:

- imitation seems to preserve a useful structure for some geometry shifts
- but that strength does not transfer uniformly to every terrain family

### Static dynamics OOD

Main result:

- Baseline 2 is the safest all-around static dynamics baseline

Notable cases:

- Baseline 1 is best on ultra-high friction
- Baseline 1 is best on heavy mass
- Baseline 2 is best on ultra-low friction
- Baseline 2 is clearly best on very weak motors
- Baseline 3 is weakest under severe actuator weakness

Interpretation:

- warm-start produced the most dependable controller under fixed hidden mismatch
- Baseline 1 remained unexpectedly sturdy on some shifts
- Baseline 3 does not dominate under static dynamics stress

### Push recovery OOD

Main result:

- Baseline 3 is the strongest first-pass push-recovery baseline

Notable cases:

- Baseline 3 is best on yaw pushes
- Baseline 3 is best on forward push recovery
- Baseline 1 is best on lateral medium push
- Baseline 1 and Baseline 3 are nearly tied on repeated lateral pushes
- Baseline 2 trails on this push suite

Interpretation:

- the imitation-shaped policy appears to recover better from transient external
  disturbances
- this is a different kind of robustness from static mismatch tolerance

### Mid-episode switch OOD

Main result:

- Baseline 1 is the strongest abrupt regime-switch baseline overall

Notable cases:

- Baseline 1 is best on switch-to-low-friction
- Baseline 1 is best on switch-to-heavy
- Baseline 1 is best on low-friction-plus-heavy switch
- Baseline 2 is best only on switch-to-very-weak-motors
- Baseline 3 is weakest overall in this switch family

Interpretation:

- the scratch controller may have retained a less specialized but more
  shock-tolerant response under sudden hidden regime changes
- Baseline 2 and especially Baseline 3 appear more specialized to their learned
  operating families

## Baseline Identities

Taken together, the frozen baselines now have fairly distinct identities.

### Baseline 1

Identity:

- honest scratch reference
- not the cleanest controller
- but surprisingly resilient under some heavy and abrupt OOD changes

Best description:

- weakest canonical controller
- best abrupt switch baseline
- scientifically valuable because it resists simple “scratch is always worst”
  narratives

### Baseline 2

Identity:

- cleanest canonical controller
- safest all-around default
- best static dynamics mismatch baseline

Best description:

- the baseline to use when the goal is a stable, defensible, deployable
  proprio-only reference

### Baseline 3

Identity:

- robustness-leaning variant shaped by temporary imitation
- strongest geometry-interesting controller
- strongest push-recovery controller
- weaker under abrupt hidden regime switches and weak actuators

Best description:

- the most behaviorally interesting variant
- not the safest universal default

## What We Can Defend Clearly

The current frozen evidence supports these claims:

- warm-start is the clearest improvement for clean blind locomotion
- temporary imitation changes the robustness profile rather than uniformly
  improving the controller
- different robustness axes produce different winners
- reward alone would have hidden these distinctions

This is already a strong technical result because it shows that:

- “robustness” is not one thing
- static mismatch tolerance, disturbance recovery, geometry shift tolerance,
  and abrupt regime-switch tolerance are separable controller properties

## Recommended Research Framing

For technical audiences, the clean framing is:

- use Baseline 2 as the canonical proprio-only reference
- use Baseline 3 as the robustness-leaning comparison variant
- keep Baseline 1 in the story because it reveals important OOD tradeoffs

That framing is stronger than pretending the ladder has a single universal
winner.

## What This Suggests For The Next Phase

The synthesis points toward a useful next design question for later teacher or
adaptation work:

> can we combine Baseline 2’s clean nominal behavior, Baseline 3’s disturbance
> recovery and geometry tolerance, and Baseline 1’s abrupt switch resilience?

That is a much sharper next-step question than simply “make the policy more
robust.”

## Bottom Line

The frozen ladder did its job.

It did not produce one baseline that wins every axis. It produced something
better:

- a clean canonical baseline
- two contrasting alternatives with distinct robustness personalities
- a concrete map of which interventions help which failure family

That gives the project a solid foundation for whatever comes next.
