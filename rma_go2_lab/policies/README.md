# RMA-Go2 Policies Archive

This directory contains the frozen policy milestones for the RMA-Go2 project.

The project follows an RMA-inspired staged structure:

- train a clean deployable flat backbone
- train a privileged rough teacher from that backbone
- refine the teacher on obstacle-heavy terrain
- distill/adapt into a deployable student later

---

## Final Teacher Lineage

The final teacher was produced through a staged warm-start chain. This is intentional and should be preserved when reproducing the project:

1. **Flat backbone expert**
   - Run: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_flat/2026-04-02_12-26-57`
   - Selected checkpoint: `model_1499.pt`
   - Role: clean deployable forward-trot prior, analogous to the prior/base policy used as a backbone in staged RMA-style pipelines.

2. **Stage B general rough RMA teacher**
   - Run: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-02_18-41-09`
   - Selected checkpoint: `model_1999.pt`
   - Frozen artifact: `rough_teacher_v1_base.pt`
   - Role: privileged rough-terrain teacher initialized from the flat backbone. This solved general rough locomotion but still showed the Stair Paradox.

3. **Stage D stair/pyramid refinement**
   - Run: `/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-06_11-18-41`
   - Warm start: `2026-04-02_18-41-09/model_1999.pt`
   - Selected checkpoint: `model_240.pt`
   - Frozen artifact: `rough_teacher_v2_refined_240.pt`
   - Role: current final privileged teacher for student distillation, refined on stair/pyramid-heavy terrain.

Important: do **not** interpret `rough_teacher_v2_refined_240.pt` as a teacher trained from scratch. It is a refinement of the Stage B rough teacher, which itself was initialized from the flat backbone expert.

Also important: `rough_teacher_v2_refined_240.pt` is **not deployable as-is**. It is a privileged teacher that uses simulation-only dynamics and terrain-height information. The deployable output of the RMA pipeline should come from the later student/adaptation phase.

---

### 1. [rough_teacher_v1_base.pt](file:///home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v1_base.pt)
*   **Stage:** Phase 3 - Stage B (Generalist)
*   **Description:** The first successful teacher trained on standard rough terrain curriculum.
*   **Performance:** Highly robust on uneven ground but suffered from the **"Stair Paradox"** (struggled with Level 9 pyramid stairs despite having sensors).
*   **Use Case:** Baseline for general rough terrain locomotion.

### 2. [rough_teacher_v2_refined_100.pt](file:///home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v2_refined_100.pt)
*   **Stage:** Phase 3 - Stage D (Specialist) - Iteration 100
*   **Description:** The first specialized checkpoint after forcing 90% stair proportions.
*   **Use Case:** Early-stage expert for student distillation debugging and comparison against the later refined checkpoint.

### 3. [rough_teacher_v2_refined_240.pt](file:///home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v2_refined_240.pt) (current final teacher)
*   **Stage:** Phase 3 - Stage D refinement - Iteration 240
*   **Description:** Current best privileged teacher artifact for Phase 2 student/adaptation experiments.
*   **Strengths:** Strong on held-out level-9 random rough, stair descent, and several dynamics-stress cases.
*   **Known weaknesses:** Level-9 stair ascent and boxes still fail heavily in the complete stress suite, so do not describe this as a fully solved max-difficulty stair-climbing policy.
*   **Use Case:** Supervisor / teacher checkpoint for student distillation and adaptation experiments.

## Current Qualification Artifacts

Final teacher manifest:

- `artifacts/manifests/final_teacher_stage_d_rough_teacher_v2_refined_240_manifest.md`
- `artifacts/manifests/final_teacher_stage_d_rough_teacher_v2_refined_240_manifest.yaml`

Complete teacher stress suite:

- `artifacts/evaluations/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999.json`
- `artifacts/evaluations/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999.csv`

Visualization:

- `artifacts/evaluations/plots/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999_survival.png`
- `artifacts/evaluations/plots/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999_tracking.png`
- `artifacts/evaluations/plots/isolated_suite_rough_teacher_v2_refined_240_complete_teacher_level9_normal_seed999_scorecard.png`

---

> [!TIP]
> Use `rough_teacher_v2_refined_240.pt` as the default teacher for Phase 2 student/adaptation work, but keep the complete stress-suite limitations visible when reporting results.
