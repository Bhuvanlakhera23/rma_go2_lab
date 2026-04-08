# rma_go2_lab/envs/student_cfg_rough_na.py
import torch
from isaaclab.utils import configclass
from rma_go2_lab.envs.teacher_cfg_rough_na import Go2RMATeacherRoughEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm

def proprioceptive_history(env, history_len: int = 30, proprio_dim: int = 48):
    """
    Returns the last N proprioceptive observations (48-dim each).
    Maintains a rolling buffer on the environment object.
    """
    # 🔴 INITIALIZATION CHECK:
    # IsaacLab calls this function once during ObservationManager.__init__ to determine dims.
    # At that point, env.observation_manager does not exist yet.
    if not hasattr(env, "observation_manager") or env.observation_manager is None:
        return torch.zeros((env.num_envs, history_len * proprio_dim), device=env.device)

    # 1. Access current raw proprioception directly to avoid CIRCULAR DEPENDENCY
    # (The "policy" group depends on "history", so we can't compute "policy" inside history)
    # We manually mirror the 48-dim policy terms from teacher_cfg_rough_na
    current_obs = torch.cat([
        mdp.base_lin_vel(env),
        mdp.base_ang_vel(env),
        mdp.projected_gravity(env),
        mdp.generated_commands(env, command_name="base_velocity"),
        mdp.joint_pos(env),
        mdp.joint_vel(env),
        mdp.last_action(env),
    ], dim=-1)

    # 2. Lazy-init the buffer on the environment
    if not hasattr(env, "_proprio_history_buffer"):
        env._proprio_history_buffer = torch.zeros(
            (env.num_envs, history_len, proprio_dim),
            device=env.device,
            dtype=current_obs.dtype
        )

    # 3. Roll and update buffer
    env._proprio_history_buffer = torch.roll(env._proprio_history_buffer, shifts=-1, dims=1)
    env._proprio_history_buffer[:, -1, :] = current_obs

    # --- AUDIT LOGS ---
    if not hasattr(env, "_history_audit_step"): env._history_audit_step = 0
    env._history_audit_step += 1
    if env._history_audit_step % 1000 == 0:
         b_mag = env._proprio_history_buffer.abs().mean().item()
         c_mag = current_obs.abs().mean().item()
         print(f"  [Env Audit] History Mag: {b_mag:.4f} | Raw Proprio Mag: {c_mag:.4f}")

    # 4. Flatten for the neural network input: [Batch, HistoryLen * ProprioDim]
    return env._proprio_history_buffer.view(env.num_envs, -1)
# 🎓 Student Adaptation: Proprioceptive History
# ------------------------------------------------
# We add a dedicated "history" group that the Student AC uses to predict z.
# DELETED HistoryGroup to avoid observation selector confusion
# (History is now part of the 'policy' group below)

@configclass
class Go2RMAStudentRoughEnvCfg(Go2RMATeacherRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        print("\n========== RMA STUDENT SPECIALIZATION \ ==========\n")

        # ------------------------------------------------
        # 🎓 Student Adaptation: Proprioceptive History
        # ------------------------------------------------
        # We inject the history directly into the primary 'policy' group.
        # This ensures it's always computed and available for the MLP.
        self.observations.policy.proprio_history = ObsTerm(
            func=proprioceptive_history,
            params={"history_len": 30}
        )

        # ------------------------------------------------
        # 🟡 Distillation Logic: Teacher Supervision
        # ------------------------------------------------
        # The Student still receives the privileged information in the "critic" group
        # (for training) and to provide the labels for the adaptation loss.
        # This is already handled by the Teacher base class.

        # ------------------------------------------------
        # 🛠️ Curriculum: Same as Teacher
        # ------------------------------------------------
        # We want the student to learn in the same high-difficulty stair environment
        # that the Stage D teacher mastered.
        self.scene.terrain.max_init_terrain_level = 5 # Start mid-difficulty
