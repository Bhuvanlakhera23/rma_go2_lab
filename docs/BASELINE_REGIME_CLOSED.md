# Baseline Regime Closed

Date closed:

- 2026-04-20

## Status

The proprio baseline regime is considered complete and frozen.

This closure includes:

- corrected flat-prior provenance
- frozen Baseline 1, Baseline 2, and Baseline 3 checkpoints
- canonical evaluation artifacts
- canonical baseline comparison note
- exploratory OOD coverage across:
  - geometry shifts
  - static dynamics shifts
  - push recovery
  - mid-episode regime switches
- matching clip artifacts for the main OOD cases
- consolidated synthesis and short-form summary notes

## Canonical Baseline Set

- `rma_go2_lab/policies/flat1499.pt`
- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`

## Governing Decision

No further baseline expansion is required for the current project phase.

Additional baseline experiments should only be added if they answer a specific
new scientific question, not as open-ended continuation of the baseline ladder.

## Current Interpretation

- Baseline 2 is the clean canonical blind baseline.
- Baseline 3 is the robustness-leaning comparison variant.
- Baseline 1 remains an important scratch and switch-resilience reference.

## What Comes Next

The next substantial phase should build on top of the frozen baseline package,
for example:

- privileged teacher design
- later adaptation work
- perception-enabled extensions

The baseline regime should now be treated as a stable reference frame for those
later branches.
