# Training SOP

This file is the short operational SOP for the active repo state.

The current project is organized around one baseline-first ladder:

1. flat prior
2. Baseline 1: rough scratch
3. Baseline 2: rough warm-start
4. Baseline 3: rough warm-start + imitation

If another document conflicts with this one, prefer this file unless we
explicitly replace it with a newer SOP.

## Goal

The goal is to build and compare a clean proprio-only locomotion ladder under a
frozen benchmark.

The repo is not currently organized around the old teacher/student training
story. That architecture has been removed from the active branch.

## Governing Principle

The purpose of the active pipeline is not:

- maximum reward
- maximum terrain difficulty
- maximum obstacle specialization

The purpose is:

- establish stable frozen baselines
- evaluate them under the same mismatch protocol
- make later comparisons interpretable

## Active Pipeline

### Stage 1: Flat Prior

Task:

- `RMA-Go2-Flat`

Purpose:

- learn a clean nominal locomotion prior
- provide the warm-start source for later baselines

Qualification rule:

- freeze one validated checkpoint
- keep only the small sanity record needed for reuse

Current frozen artifact:

- `rma_go2_lab/policies/flat1499.pt`

### Stage 2: Baseline 1

Task:

- `RMA-Go2-Blind-Baseline-Rough`

Purpose:

- establish the honest scratch rough baseline

Qualification rule:

- train with the frozen shared rough task
- evaluate with the frozen suite and gait checks
- freeze exactly one checkpoint and its clip set

Current frozen artifact:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`

### Stage 3: Baseline 2

Task:

- `RMA-Go2-Blind-Baseline-Rough-WarmStart`

Purpose:

- measure the value of flat-prior warm-start over scratch

Qualification rule:

- same shared rough task as B1
- only the initialization changes
- evaluate with the same frozen suite and clip manifest

Current frozen artifact:

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`

### Stage 4: Baseline 3

Task:

- `RMA-Go2-Blind-Baseline-Rough-WarmStart-Imitation`

Purpose:

- measure whether temporary imitation from the flat prior improves over B2

Qualification rule:

- same shared rough task as B1/B2
- same frozen evaluation protocol
- only the training mechanism changes

## Fairness Rules

When comparing B1, B2, and B3:

- keep the rough environment shared
- keep the actor observation interface fixed
- keep the command distribution fixed
- keep the evaluation suite fixed
- keep the recording manifest fixed
- change only the intended experimental variable

Do not casually broaden commands, change rewards, or change terrain families in
only one baseline.

## Canonical Evaluation

The active evaluation stack is:

- `scripts/eval/gait.py`
- `scripts/eval/run_isolated_suite.py`
- `scripts/eval/record_clip.py`
- `scripts/eval/record_overview_clip.py`

Primary frozen suite:

- `blind_baseline_v1`

Primary artifact families:

- `artifacts/evaluations/flat_prior/`
- `artifacts/evaluations/baseline1/`
- `artifacts/evaluations/baseline2/`
- `artifacts/evaluations/baseline3/`
- `artifacts/evaluations/clips/`

## Canonical Recording Manifest

Every frozen blind baseline should carry the same clip set:

- `nominal_overview`
- `nominal_hero`
- `low_friction_hero`
- `weak_motor_hero`
- `heavy_mass_hero`

Keep recording conditions identical across B1, B2, and B3. Only the checkpoint
should change.

Store clips under:

- `artifacts/evaluations/clips/baseline1/`
- `artifacts/evaluations/clips/baseline2/`
- `artifacts/evaluations/clips/baseline3/`

Each clip folder should contain:

- one `.mp4`
- one `metadata.json`

## Freeze Rules

A run is ready to freeze only when all of the following are true:

1. the training run reached its intended horizon or a justified selected
   checkpoint
2. gait evals are recorded
3. the frozen suite is complete, not partial
4. the canonical clip set is recorded
5. the checkpoint is copied into `rma_go2_lab/policies/`
6. a freeze note is written under `rma_go2_lab/policies/`

Do not silently replace a frozen checkpoint in place.

If a better version is trained later:

- create a new explicitly versioned artifact
- do not overwrite the old one

## What To Commit

Commit:

- code
- docs
- frozen `.pt` policy artifacts

Do not treat raw local run outputs as source code.

By default, do not commit:

- raw intermediate eval dumps unless they are part of the frozen artifact set
- generated temp eval folders
- ad hoc video recordings not selected for comparison
- IsaacLab log directories

## Reading Order

For repo orientation, read in this order:

1. `docs/PROJECT_GUIDE.md`
2. `TRAINING_SOP.md`
3. `rma_go2_lab/policies/blind_baseline_protocol.md`
4. `rma_go2_lab/policies/README.md`
5. `artifacts/evaluations/README.md`

## One-Line Mental Model

This repo is a clean proprio-baseline ladder:

train one flat prior, freeze B1/B2/B3 under a shared rough benchmark, and keep
the artifacts structured enough that comparisons stay obvious.
