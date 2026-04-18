# Blind Baseline Protocol

This document is the canonical SOP for all blind-baseline work in the RMA-Go2
project.

If a future training change conflicts with this document, follow this document
unless we explicitly replace it with a newer protocol.

## Governing Principle

The goal of the blind baseline is not to maximize terrain difficulty, maximize
reward, or maximize survival at all costs.

The goal is to measure how a fixed proprioceptive locomotion policy degrades
under increasing unobservable environment mismatch.

Examples of mismatch:

- friction variation
- base-mass variation
- motor-strength variation
- terrain geometry variation
- mid-episode dynamics changes

The blind baseline should fail gracefully and informatively, leaving room for
adaptation methods such as RMA to show why they are needed.

## What The Blind Baseline Is

The blind baseline is:

- a deployable-style proprioceptive policy
- symmetric actor-critic during training
- no privileged observations
- no latent `z`
- no teacher encoder path

Current experiment ladder:

1. `RMA-Go2-Blind-Baseline-Rough`
   scratch blind rough baseline
2. `RMA-Go2-Blind-Baseline-Rough-WarmStart`
   same blind rough baseline, actor warm-started from the validated flat prior
3. `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation`
   same blind rough baseline, warm-start plus temporary imitation prior

The flat prior is not Baseline 1. It is a locomotion prior used to initialize
Baseline 2 and Baseline 3.

## What The Blind Baseline Is Not

The blind baseline is not:

- a max-difficulty obstacle specialist
- a stair-climbing benchmark by default
- a reward-maximization project
- a proxy for the privileged teacher
- proof that RMA is unnecessary

Do not harden the blind baseline so much that the adaptation gap disappears for
artificial reasons.

## Training Policy

Training should produce a competent fixed controller, not the strongest
possible blind locomotion policy under every imaginable condition.

Rules:

- Keep the actor observation interface fixed.
- Keep the command distribution fixed unless there is a very strong reason to
  change it for all baselines and RMA.
- Prefer a simple forward-command regime for the blind comparison ladder.
- Do not keep expanding terrain families during baseline training just because
  the policy survives them.
- Do not introduce privileged signals, history latents, or hidden teacher
  shortcuts into blind baselines.
- Avoid reward inflation and overlapping shaping unless a concrete pathology
  requires it.

## Canonical Baseline-2 Intent

`RMA-Go2-Blind-Baseline-Rough-WarmStart` is a general rough warm-start
baseline.

It is intended to answer:

- how much a clean flat locomotion prior helps rough locomotion
- what failure modes remain without privileged adaptation

It is not intended to answer:

- what is the strongest obstacle-specialized blind policy we can train
- how well a blind policy can overfit to stairs or boxes

Therefore:

- mixed rough terrain is acceptable
- moderate curriculum is acceptable
- stairs and boxes are not part of the default baseline-training mix

If stairs or boxes are used later, they should first appear in evaluation, not
baseline training.

## Evaluation Principle

Evaluation is where mismatch should become explicit.

All baselines and future RMA methods should be compared under the same fixed
evaluation protocol.

Primary evaluation axes:

- friction sweep
- mass sweep
- motor-strength sweep
- terrain-level sweep
- terrain-family sweep
- mid-episode mismatch changes

Primary metrics:

- velocity tracking error
- time-to-failure
- failure-cause distribution
- recovery time after mismatch change
- slip and drift
- body-height and orientation instability
- action effort / energy proxy

Secondary metrics:

- reward
- success or survival rate

Reward is a training signal. It is not the final scientific metric.

Current frozen evaluation entrypoints:

- `scripts/eval/run_isolated_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --suite blind_baseline_v1`
- `scripts/eval/gait.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --command-profile standstill`
- `scripts/eval/gait.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --command-profile forward`

The current `blind_baseline_v1` suite is the canonical static-mismatch suite.
Mid-episode mismatch evaluation is still a later extension and must be added as
an explicit new versioned suite rather than replacing `blind_baseline_v1`
informally.

## Canonical Recording Manifest

Frozen baselines should also have a small, fixed video artifact set.

The purpose of recorded clips is not to exhaustively show every randomization
combination. The purpose is to provide a compact visual record of:

- nominal multi-environment behavior
- nominal single-robot behavior
- the most important benchmark-aligned stress cases

The recording manifest must stay the same across:

- `RMA-Go2-Blind-Baseline-Rough`
- `RMA-Go2-Blind-Baseline-Rough-WarmStart`
- `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation`

Canonical clip set:

- `nominal_overview`
- `nominal_hero`
- `low_friction_hero`
- `weak_motor_hero`
- `heavy_mass_hero`

Definitions:

- `nominal_overview`
  overview clip with multiple environments visible; use fixed env-index
  sampling rather than ad hoc manual camera movement
- `nominal_hero`
  single-environment clip for close inspection of the selected baseline
- `low_friction_hero`
  single-environment stress clip under a fixed low-friction override
- `weak_motor_hero`
  single-environment stress clip under a fixed reduced motor-strength override
- `heavy_mass_hero`
  single-environment stress clip under a fixed positive mass-offset override

Recording rules:

- keep clip conditions identical across B1, B2, and B3
- change only the checkpoint when comparing baselines
- keep clip length consistent unless a new protocol version says otherwise
- keep overview env-index sampling consistent unless a new protocol version says
  otherwise
- treat the recording manifest as a comparison artifact, not a highlight reel

Current recording tools:

- `scripts/eval/record_clip.py`
- `scripts/eval/record_overview_clip.py`

Storage layout:

- flat prior clips:
  `artifacts/evaluations/clips/flat_prior/`
- Baseline 1 clips:
  `artifacts/evaluations/clips/baseline1/`
- Baseline 2 clips:
  `artifacts/evaluations/clips/baseline2/`
- Baseline 3 clips:
  `artifacts/evaluations/clips/baseline3/`

Recommended current usage pattern:

- `nominal_overview`
  `record_overview_clip.py` with `--num_envs 100`, fixed sampled indices such
  as `0,30,60,90`, and a fixed `--switch_interval`
- `nominal_hero`
  `record_clip.py` with `--num_envs 1`
- stress hero clips
  `record_clip.py` with `--num_envs 1` plus one explicit override family at a
  time

When recording canonical clips, set `--output_dir` to the artifact-specific
clip folder rather than relying on the shared default clip root.

Do not create the canonical frozen artifact set from arbitrary combinations of:

- friction
- terrain type
- terrain level
- mass
- motor strength

If the recording manifest changes later, version it explicitly instead of
silently replacing these conditions.

## When To Stop Retuning

Do not restart training repeatedly because the policy is not yet perfect.

Treat a baseline configuration as stable once:

- command distribution is fixed
- observation interface is fixed
- reward structure is coherent and not obviously pathological
- termination structure is coherent
- training curve has been allowed to plateau

At that point, improvement should come from:

- evaluation and diagnosis
- controlled ablations
- explicitly versioned successor baselines

not from endless informal retuning of the same run family.

## How To Validate A Ceiling

Do not call something a ceiling from one training curve alone.

A candidate ceiling becomes credible only when all three are true:

1. plateau:
   training metrics flatten over a long window
2. controlled perturbation:
   small reward or termination tweaks do not materially change the qualitative
   outcome
3. held-out evaluation:
   the failure mode is stable across mismatch sweeps

If tiny reward changes produce large differences, that is a training-design
ceiling, not a policy/environment ceiling.

## Reward Design Policy

Default philosophy for blind baselines:

- tracking-dominant
- light regularization
- enough terrain/stability shaping to make training viable
- no excessive gait scripting unless explicitly justified

Use other repos for selective ideas only.

Do not merge reward philosophies indiscriminately from:

- `quadrupeds_locomotion`
- `unitree_rl_gym`
- `walk-these-ways`
- `rl_locomotion`

because that creates an over-engineered baseline with unclear attribution.

## Decision Rules For Future Changes

Before changing a blind baseline training config, answer:

1. Does this make the blind policy more interpretable under mismatch?
2. Does this preserve fair comparison with future RMA experiments?
3. Is this change needed to remove a concrete pathology, or are we only chasing
   a higher reward?

If the answer to `3` is only "higher reward", do not change the baseline.

## Current Project Commitment

From this point onward:

- Baseline 1, 2, and 3 are comparison baselines, not moving targets
- mismatch-focused evaluation is the main lens for judging them
- future RMA work must be justified by blind-baseline failure under hidden
  mismatch, not by vague intuition

This is the project stonemarker for blind-baseline work.
