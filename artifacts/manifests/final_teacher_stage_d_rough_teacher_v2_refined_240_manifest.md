# Final Teacher Manifest: rough_teacher_v2_refined_240

Generated: 2026-04-08T12:53:27

## Status

- stage: Stage D final teacher refinement
- selected artifact: `rma_go2_lab/policies/rough_teacher_v2_refined_240.pt`
- sha256: `c2e71287374a956eac76fb022894ccd72a666558242bf03fdaba632d1fd1943c`
- status: `golden_teacher_for_student_distillation`
- warning: rough_teacher_v2_refined_240.pt is not a from-scratch teacher. It is a stair/pyramid refinement of the Stage B general rough teacher, which was initialized from the flat backbone.

## Lineage

- Stage A flat backbone: deployable proprioceptive locomotion prior analogous to policy_22000.pt in the reference RMA-style pipeline
  source checkpoint: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_flat/2026-04-02_12-26-57/model_1499.pt`
  notes: Selected after visual qualification; diagonal trot-like gait FR+RL then FL+RR.
- Stage B general rough RMA teacher: privileged rough teacher initialized from flat backbone
  source checkpoint: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-02_18-41-09/model_1999.pt`
  frozen artifact: `rma_go2_lab/policies/rough_teacher_v1_base.pt` sha256 `54844e158e78e9c46489d4f37262ceb58f0e9a4cce35123130f4e562ce941d08`
- Stage D stair/pyramid refinement: refinement run emphasizing stair up/down and boxes
  source checkpoint: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-06_11-18-41/model_240.pt`
  source run: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-06_11-18-41`
  frozen artifact: `rma_go2_lab/policies/rough_teacher_v2_refined_240.pt` sha256 `c2e71287374a956eac76fb022894ccd72a666558242bf03fdaba632d1fd1943c`

## Architecture

- policy_observation_dim: `48`
- privileged_observation_dim: `204`
- actor_input_dim: `56`
- critic_input_dim: `252`
- latent_dim: `8`
- action_dim: `12`
- actor_hidden_dims: `[512, 256, 128]`
- critic_hidden_dims: `[512, 256, 128]`
- teacher_deployability: `not deployable as-is; teacher uses privileged dynamics + terrain height scan. Distill/adapt into student before deployment.`

## Evaluation Summary

- complete suite status: `complete` (16/16)
- num_envs: `64`
- steps: `1000`
- seed: `999`
- JSON: `artifacts/evaluations/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999.json`
- CSV: `artifacts/evaluations/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999.csv`
- plot: `artifacts/evaluations/plots/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999_survival.png`
- plot: `artifacts/evaluations/plots/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999_tracking.png`
- plot: `artifacts/evaluations/plots/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999_scorecard.png`

## Key Findings

- nominal_random_rough_l9: score `15.79`, timeouts `57`, base contacts `9`, vel_err `0.213`, yaw_err `0.201`
- nominal_stairs_down_l9: score `18.37`, timeouts `63`, base contacts `1`, vel_err `0.131`, yaw_err `0.127`
- nominal_stairs_up_l9: score `-0.60`, timeouts `18`, base contacts `88`, vel_err `0.209`, yaw_err `0.140`
- nominal_boxes_l9: score `7.82`, timeouts `34`, base contacts `45`, vel_err `0.189`, yaw_err `0.119`

## Interpretation

- Strong on random_rough level 9, stair descent, and several dynamics extremes.
- Weak on nominal stairs-up level 9 and boxes level 9; do not claim the teacher is fully stair-ascent-qualified at max difficulty.
- Latent ablation is not conclusive for stair-up because all stair-up variants fail heavily.

## Config Files

- `rma_go2_lab/envs/teacher_cfg_rough_na.py`
- `rma_go2_lab/models/teacher_ppo_cfg_na.py`
- `rma_go2_lab/models/rma_actor_critic.py`
- `rma_go2_lab/models/ppo_with_rma_adaptation.py`

