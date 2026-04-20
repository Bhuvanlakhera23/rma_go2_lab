# OOD Evaluation Scripts

This directory is reserved for exploratory out-of-distribution evaluation
helpers.

Use it for:

- OOD suite launchers
- scenario manifests
- geometry or dynamics stress-test wrappers
- comparison utilities that are intentionally non-canonical

Do not put OOD-specific helpers into `scripts/eval/` unless they become part of
the canonical evaluation protocol later.

## Current Files

- `run_ood_suite.py`
  - launch a named OOD scenario list across frozen baselines
- `ood_scenarios.py`
  - explicit scenario manifest for stairs, boxes, harsher roughness, and
    combined dynamics mismatch, plus push-recovery and switch probes

Current suite names:

- `ood_geometry_v1`
- `ood_dynamics_v1`
- `ood_combo_v1`
- `ood_push_v1`
- `ood_switch_v1`
- `ood_limit_v1`

## Example

Run the first geometry OOD sweep on Baseline 2:

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt --suite ood_geometry_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/baseline2
```

Run the first push-recovery OOD sweep on Baseline 2:

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt --suite ood_push_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/baseline2
```

Run the first mid-episode switch OOD sweep on Baseline 2:

```bash
env TERM=xterm ./isaaclab.sh -p /home/bhuvan/projects/rma/rma_go2_lab/scripts/eval_ood/run_ood_suite.py --task RMA-Go2-Blind-Baseline-Rough-WarmStart --checkpoint /home/bhuvan/projects/rma/rma_go2_lab/rma_go2_lab/policies/blind_baseline2_warmstart_final.pt --suite ood_switch_v1 --output-dir /home/bhuvan/projects/rma/rma_go2_lab/artifacts/ood_evaluations/baseline2
```

## Current Rule

Any script added here should:

- write to `artifacts/ood_evaluations/`
- clearly label outputs as exploratory / OOD
- avoid reusing canonical output filenames
