# 🐾 RMA-Go2 Policies Archive

This directory contains the key trained teacher models for the RMA-Go2 project. Each file represents a milestone in the robot's journey from general rough terrain locomotion to high-difficulty obstacle mastery.

---

### 1. [rough_teacher_v1_base.pt](file:///home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v1_base.pt)
*   **Stage:** Phase 3 - Stage B (Generalist)
*   **Description:** The first successful teacher trained on standard rough terrain curriculum.
*   **Performance:** Highly robust on uneven ground but suffered from the **"Stair Paradox"** (struggled with Level 9 pyramid stairs despite having sensors).
*   **Use Case:** Baseline for general rough terrain locomotion.

### 2. [rough_teacher_v2_refined_100.pt](file:///home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v2_refined_100.pt)
*   **Stage:** Phase 3 - Stage D (Specialist) - Iteration 100
*   **Description:** The first specialized checkpoint after forcing 90% stair proportions.
*   **Fixes:** Resolved the Stair Paradox. This model is **1.8x safer** on stairs when sensors are active than when it is blind.
*   **Use Case:** Early-stage expert for student distillation debugging.

### 3. [rough_teacher_v2_refined_240.pt](file:///home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/rough_teacher_v2_refined_240.pt) (🏆 THE GOLDEN EXPERT)
*   **Stage:** Phase 3 - Stage D (Peak Performance) - Iteration 240
*   **Description:** The definitive "Stair Master" and rough terrain expert. 
*   **Metrics:** **2.24% fall rate** and **0.053 tracking error** on max-difficulty stairs.
*   **Use Case:** **SUPERVISOR for student distillation.** This model strikes the perfect balance between perceptive sensitivity and raw survival.

---

> [!TIP]
> Use `rough_teacher_v2_refined_240.pt` for all distillation runs unless you are specifically testing for robustness on non-stair terrain, in which case `rough_teacher_v1_base.pt` provides a more "relaxed" gait.
