# Training SOP

This file defines the standard operating procedure for training and qualifying the Go2 RMA pipeline in this repository.

The goal is to stop ad-hoc trial-and-error and replace it with staged training, explicit entry and exit criteria, and a repeatable qualification workflow.

## Principles

- Train one difficulty axis at a time.
- Do not widen the task and increase robustness stressors in the same stage.
- Treat scalar reward as necessary but not sufficient.
- Visual inspection is mandatory for checkpoint qualification.
- A stage is complete only when its exit criteria are met.
- Do not start student training until the teacher is explicitly qualified.

## Canonical Artifacts

For every serious run, keep:

- the IsaacLab run directory under `logs/rsl_rl/...`
- one manifest from `scripts/export_teacher_manifest.py`
- one checkpoint evaluation report from `scripts/eval_teacher.py`

Do not copy large checkpoint binaries into this repo.
Keep the source of truth for checkpoints in the IsaacLab log directories and record the selected paths in manifests or reports.

## Stage Definitions

### Stage A: Flat Seed

Purpose:

- learn a stable forward locomotion prior
- produce the bootstrap checkpoint for the rough teacher

Allowed task complexity:

- flat terrain only
- forward-only commands
- no pushes

Allowed robustness/randomization:

- moderate mass/friction/gain randomization is acceptable
- do not expand the task to rough terrain or obstacle negotiation here

Minimum acceptance criteria:

- episodes consistently time out on flat terrain
- near-zero `base_contact`
- straight forward walking in playback
- no obvious drift, shuffling, or freezing

Deliverable:

- one canonical flat checkpoint path

### Stage B: Rough Teacher Warm Start

Purpose:

- transfer the flat gait to rough terrain
- learn terrain-conditioned behavior using privileged information

Required setup:

- bootstrap from the canonical flat checkpoint
- forward-only commands
- monotonic terrain curriculum
- no pushes
- no command-space expansion beyond what the flat seed supports

Current intended configuration:

- actor uses policy obs plus latent `z`
- encoder uses privileged terrain and dynamics
- critic uses policy obs plus privileged obs

Allowed robustness/randomization:

- friction randomization
- mass randomization
- motor stiffness and damping randomization
- no rollout pushes yet

What this stage is not:

- not stair-climbing specialization
- not multidirectional control
- not the student stage

Exit criteria:

- rough terrain levels increase and remain high without curriculum collapse
- `base_contact` is materially reduced relative to early training
- playback shows stable rough forward walking
- latent ablation matters:
  - normal `z` performs best
  - zeroed `z` degrades
  - shuffled `z` degrades

Deliverable:

- one best Stage B checkpoint selected by evaluation, not by last iteration alone

### Stage C: Rough Robustness

Purpose:

- improve robustness on harder rough terrain while preserving the Stage B gait prior

Input:

- best Stage B checkpoint

Allowed changes:

- increase terrain difficulty
- increase breadth of randomized dynamics gradually
- introduce observation noise and latency if deployment requires them

Do not change yet:

- no pushes unless Stage D is intentionally started
- no broad command-space expansion

Exit criteria:

- stable playback on harder rough terrain
- lower failure rate than Stage B on the same held-out terrains
- no collapse to conservative near-static behavior

Deliverable:

- one qualified rough forward teacher checkpoint

### Stage D: Obstacle Curriculum

Purpose:

- teach behaviors that generic rough terrain does not reliably induce
- especially step-up, curb, stair-up, and stair-down traversal

Rationale:

- generic rough-terrain success does not imply stair-climbing success
- descending obstacles is easier than ascending them
- if the robot can descend but cannot climb the first step, this stage is missing

Allowed changes:

- introduce dedicated obstacle terrains
- use staged obstacle difficulty
- add small task shaping only if behavior clearly fails without it

Preferred order of intervention:

1. dedicated obstacle terrains
2. targeted evaluation on those terrains
3. only then consider small shaping terms for obstacle clearance or repeated collision failure modes

Do not do:

- do not add many new rewards at once
- do not assume more training time alone will create step-up behavior

Exit criteria:

- robot can climb the first step from flat ground
- robot can negotiate repeated step-ups in playback
- robot can descend without collapse
- latent-conditioned behavior visibly changes before obstacle contact

Deliverable:

- one obstacle-capable teacher checkpoint

### Stage E: Disturbance Hardening

Purpose:

- improve recovery robustness after the base locomotion and obstacle behaviors already exist

Allowed changes:

- rollout pushes
- stronger disturbance events
- possibly re-enable additional body randomization such as base COM if needed

Input:

- best pre-push teacher checkpoint

Rules:

- keep the pre-push checkpoint as fallback
- do not introduce pushes before the base teacher is already qualified

Exit criteria:

- recovery improves under disturbances without destroying tracking or obstacle performance

Deliverable:

- hardened teacher checkpoint

### Stage F: Student / Adaptation

Purpose:

- replace privileged terrain/dynamics encoding with an adaptation module that infers context from deployable observations

Prerequisite:

- a qualified teacher already exists

Rules:

- do not train the student against an unqualified teacher
- evaluate student against the same terrain and obstacle suites used for teacher qualification

Deliverable:

- deployable student using proprioceptive history rather than privileged observations

## Qualification Workflow

Every serious checkpoint sweep should include:

1. Scalar review
- reward
- episode length
- `base_contact`
- `time_out`
- `terrain_levels`
- `error_vel_xy`
- `error_vel_yaw`
- latent statistics if available

2. Latent ablation
- normal `z`
- zeroed `z`
- shuffled `z`

3. Visual playback
- flat terrain
- held-out rough terrain
- obstacle terrains if the stage includes them

4. Failure-mode notes
- yaw drift
- toe scuffing
- inability to climb the first step
- conservative shuffling
- terrain collapse

Use `scripts/eval_teacher.py` for scalar and ablation ranking, then confirm finalists by playback.

## Evaluation Rules

- Do not crown the final iteration automatically.
- Rank checkpoints from the run.
- Prefer a checkpoint with better behavior over a checkpoint with slightly higher scalar reward.
- If reward improves because curriculum collapsed, reject that checkpoint.
- If playback contradicts the logs, trust playback.

## Randomization Policy

Use staged realism.

Start earlier:

- friction
- mass
- actuator gain variation

Add later:

- observation noise
- command delay or latency
- pushes
- extra body perturbations such as base COM if needed

Rule:

- if a realism source makes the current stage stop learning the core skill, move it to a later stage

## Visual Inspection SOP

Do not develop locomotion headless-only.

Minimum playback checks:

- straight forward walking on flat terrain
- rough terrain traversal
- first-step ascent test
- stair descent
- yaw drift check

Reason:

- many locomotion failures are not visible in reward curves alone

## Record-Keeping SOP

For each canonical run:

- export a manifest near the start of training
- keep the run directory
- record the chosen checkpoint path
- record why the checkpoint was selected
- note known failure modes

Suggested naming:

- manifests: `<task>_<stage>_<timestamp>_manifest.*`
- evaluation reports: `<task>_<stage>_<timestamp>_eval.*`

## Current Project Status

As of now:

- Stage A is complete with the current flat checkpoint
- Stage B is the active teacher stage
- Stage D is not yet implemented as an explicit obstacle curriculum
- student training should wait until a Stage B or later teacher is explicitly qualified

## Push / Commit Rule

Push the code when a training revision reaches a coherent decision point:

- the run clearly fails and a new revision is required
- or a checkpoint is selected as the canonical output of that stage

Do not push mid-run just because the code compiles.
Push when the code and the training outcome together form a coherent stage result.
