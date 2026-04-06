# rma_go2_lab/envs/student_cfg_rough_na.py
import torch
from isaaclab.utils import configclass
from rma_go2_lab.envs.teacher_cfg_rough_na import Go2RMATeacherRoughEnvCfg
import rma_go2_lab.envs.teacher_cfg_rough_na as teacher_mdp

def proprioceptive_history(env, history_len: int = 30):
    """
    Returns the last N proprioceptive observations (48-dim each).
    Maintains a rolling buffer on the environment object.
    """
    # 1. Access current proprioception (concatenate policy obs)
    # This matches the 'policy' group definition in UnitreeGo2RoughEnvCfg
    current_obs = env.observation_manager.compute_group("policy")
    proprio_dim = current_obs.shape[-1] # Should be 48
    
    # 2. Lazy-init the buffer on the environment
    if not hasattr(env, "_proprio_history_buffer"):
        env._proprio_history_buffer = torch.zeros(
            (env.num_envs, history_len, proprio_dim), 
            device=env.device, 
            dtype=current_obs.dtype
        )
    
    # 3. Roll and update buffer
    # We shift all elements left and put the newest at the end
    env._proprio_history_buffer = torch.roll(env._proprio_history_buffer, shifts=-1, dims=1)
    env._proprio_history_buffer[:, -1, :] = current_obs
    
    # 4. Flatten for the neural network input: [Batch, HistoryLen * ProprioDim]
    return env._proprio_history_buffer.view(env.num_envs, -1)

@configclass
class Go2RMAStudentRoughEnvCfg(Go2RMATeacherRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        print("\n========== RMA STUDENT SPECIALIZATION \ ==========\n")

        # ------------------------------------------------
        # 🎓 Student Adaptation: Proprioceptive History
        # ------------------------------------------------
        # We add a dedicated "history" group that the Student AC uses to predict z.
        self.observations.history = configclass()(
            {
                "proprio_history": teacher_mdp.ObsTerm(
                    func=proprioceptive_history,
                    params={"history_len": 30}
                )
            }
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
