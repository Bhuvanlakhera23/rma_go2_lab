#!/bin/bash
CHECKPOINT="/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-02_18-41-09/model_1999.pt"

echo "Starting rigorous Per-Terrain Stress Test..."
echo "This will test Level 9 difficulty on isolated obstacles."

for TERRAIN in pyramid_stairs boxes random_rough; do
  for MODE in normal zero; do
    echo "Evaluating ${TERRAIN} in ${MODE} mode..."
    /home/bhuvan/tools/IsaacLab/isaaclab.sh -p scripts/eval_per_terrain.py \
      --checkpoint "$CHECKPOINT" \
      --terrain-type "$TERRAIN" \
      --latent-mode "$MODE" \
      --steps 500 --seed 999 --num_envs 64 --headless > stress_test_${TERRAIN}_${MODE}.txt 2>&1
  done
done

echo "Stress test complete."
