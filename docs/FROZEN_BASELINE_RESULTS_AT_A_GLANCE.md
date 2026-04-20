# Frozen Baseline Results At A Glance

This is the compact summary version of the frozen baseline story.

Use this file when you need:

- a slide-ready comparison
- a short report summary
- a quick orientation before reading the full notes

Detailed sources:

- `docs/BASELINE_COMPARISON_FINAL.md`
- `docs/OOD_FINDINGS_B1_B2_B3.md`
- `docs/FROZEN_BASELINE_SYNTHESIS.md`

## Frozen Checkpoints

| Baseline | Identity | Frozen checkpoint |
| --- | --- | --- |
| B1 | scratch blind rough baseline | `rma_go2_lab/policies/blind_baseline1_scratch_final.pt` |
| B2 | warm-start blind rough baseline | `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt` |
| B3 | warm-start + temporary imitation | `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt` |

## Canonical In-Distribution Ranking

| Rank | Baseline | Why |
| --- | --- | --- |
| 1 | B2 | cleanest locomotion, lowest drift, strongest canonical behavior quality |
| 2 | B3 | best robustness-leaning canonical variant, but messier than B2 |
| 3 | B1 | honest scratch reference |

## Main Intervention Effects

| Transition | Main gain | Main cost |
| --- | --- | --- |
| B1 -> B2 | large improvement in gait quality and drift control | none of similar magnitude |
| B2 -> B3 | better robustness balance and stronger diagonal tendency | more drift, less clean motion |

## OOD Winners By Axis

| OOD axis | Best baseline | Short interpretation |
| --- | --- | --- |
| Geometry OOD | B3 | strongest on stairs-down and boxes |
| Static dynamics OOD | B2 | safest all-around under fixed hidden mismatch |
| Push recovery OOD | B3 | strongest disturbance-recovery behavior |
| Mid-episode switch OOD | B1 | strongest abrupt regime-switch resilience |

## Baseline Identities

| Baseline | Best one-line description |
| --- | --- |
| B1 | weakest canonically, but surprisingly resilient to abrupt regime switches |
| B2 | clean canonical proprio-only reference |
| B3 | most behaviorally interesting robustness-leaning variant |

## What We Can Defend

- Warm-start is the clearest improvement for clean blind locomotion.
- Temporary imitation changes the robustness profile rather than universally improving the controller.
- Different robustness axes have different winners.
- Reward alone would have hidden these distinctions.

## Recommended Research Framing

Use:

- **B2** as the canonical blind baseline
- **B3** as the robustness-leaning comparison variant
- **B1** as the scratch and switch-resilience reference

Avoid claiming:

- that one baseline wins every axis
- that canonical and OOD results tell the same story
- that reward alone determines the better controller

## Bottom Line

The frozen ladder did not produce one universal winner.

It produced a more useful result:

- B2 is the clean baseline
- B3 is the geometry/push-recovery variant
- B1 is the abrupt-switch-resilient reference

That gives the project a real map of controller tradeoffs, not just a single ranking.
