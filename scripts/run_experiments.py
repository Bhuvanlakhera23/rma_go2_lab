import subprocess
import shlex

CHECKPOINT = "/home/bhuvan/tools/IsaacLab/logs/rsl_rl/go2_rma_teacher_na/2026-04-02_18-41-09/model_1999.pt"
tests = [
    ("pyramid_stairs", "zero"),
    ("boxes", "normal"),
    ("boxes", "zero")
]

for terrain, mode in tests:
    cmd = f"/home/bhuvan/tools/IsaacLab/isaaclab.sh -p scripts/eval_per_terrain.py --checkpoint {CHECKPOINT} --terrain-type {terrain} --latent-mode {mode} --steps 500 --seed 999 --num_envs 64 --headless"
    print(f"Running: {terrain} ({mode})")
    
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    with open(f"final_res_{terrain}_{mode}.txt", "w") as f:
        f.write(f"RETURN CODE: {result.returncode}\n")
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
        
    print(f"Finished {terrain} ({mode}) with code {result.returncode}")
