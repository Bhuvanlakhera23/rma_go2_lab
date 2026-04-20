# RMA-Go2 Policy Archive

This directory is now baseline-first and intentionally small:

- current blind-baseline procedure docs live here
- only the `.pt` files needed for current baseline comparison are kept here
- old teacher/student bundles are not part of the active workflow

Read these first:

1. `docs/PROJECT_GUIDE.md`
2. `rma_go2_lab/policies/blind_baseline_protocol.md`

## Active Meaning

Right now, the live project focus is:

1. flat prior sanity
2. blind baseline training
3. blind baseline evaluation under the frozen SOP

So this directory should not be read as proof that the teacher/student pipeline is
currently active or finalized. Those ideas remain future work in the cleaned-up
repo state.

## Canonical `.pt` Files

These are the only policy checkpoints currently kept here on purpose:

- `rma_go2_lab/policies/flat1499.pt`
  - selected flat locomotion prior
  - used to warm-start the blind warm-start baseline
  - corrected to the non-normalized `2026-04-17_14-14-36/model_1499.pt`
    lineage after restored-run audit
- `rma_go2_lab/policies/blind_baseline1_scratch_final.pt`
  - frozen final Baseline 1 checkpoint
  - selected from `model_1999.pt`
- `rma_go2_lab/policies/blind_baseline2_warmstart_final.pt`
  - frozen final Baseline 2 checkpoint
  - selected from `model_1500.pt`
- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.pt`
  - frozen final Baseline 3 checkpoint
  - selected from `model_560.pt` after an intra-run checkpoint sweep

Baseline 1 freeze note:

- `rma_go2_lab/policies/blind_baseline1_scratch_final.md`

Baseline 2 freeze note:

- `rma_go2_lab/policies/blind_baseline2_warmstart_final.md`

Baseline 3 freeze note:

- `rma_go2_lab/policies/blind_baseline3_warmstart_imitation_final.md`

Flat prior freeze note:

- `rma_go2_lab/policies/flat_prior_final.md`

## Current Evaluation Artifacts

Use `artifacts/evaluations/` for active outputs, organized by artifact family.

The flat-prior sanity artifacts currently worth keeping are:

- `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_standstill.json`
- `artifacts/evaluations/flat_prior/gait_flat_prior_model1499_forward.json`

The current blind baseline artifacts also live there as they are produced, for example:

- `artifacts/evaluations/baseline1/gait_blind_scratch_model1999_standstill.json`
- `artifacts/evaluations/baseline1/gait_blind_scratch_model1999_forward.json`
- `artifacts/evaluations/baseline1/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline1/isolated_suite_model_1999_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- `artifacts/evaluations/baseline2/gait_blind_warmstart_model1500_standstill.json`
- `artifacts/evaluations/baseline2/gait_blind_warmstart_model1500_forward.json`
- `artifacts/evaluations/baseline2/isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline2/isolated_suite_model_1500_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
- `artifacts/evaluations/baseline3/gait_blind_warmstart_imitation_model560_standstill.json`
- `artifacts/evaluations/baseline3/gait_blind_warmstart_imitation_model560_forward.json`
- `artifacts/evaluations/baseline3/isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999.json`
- `artifacts/evaluations/baseline3/isolated_suite_model_560_blind_baseline_v1_random_rough_levelspread_normal_seed999.csv`
