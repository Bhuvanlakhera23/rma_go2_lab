#rma_go2_lab/models/rma_actor_critic.py
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.networks.mlp import MLP


class RMALatentEncoder(nn.Module):
    def __init__(self, input_dim=204, latent_dim=8):
        """
        Segmented Encoder: Splits the high-dim privileged vector into terrain and dynamics.
        Assume last 187 dims are terrain (Unitree default), first (N-187) are dynamics.
        """
        super().__init__()

        self.terrain_dim = 187
        self.dynamics_dim = input_dim - self.terrain_dim

        # Branch 1: Terrain context (Geometry)
        self.terrain_branch = nn.Sequential(
            nn.Linear(self.terrain_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )

        # Branch 2: Dynamics context (Physics)
        self.dynamics_branch = nn.Sequential(
            nn.Linear(self.dynamics_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )

        # Final Bottleneck Head
        self.head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
            nn.Tanh(), # 🔴 Constrain latent space to [-1, 1]
        )

        # Orthogonal init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [Batch, TotalPrivileged]
        t_obs = x[:, -self.terrain_dim:]  # Last 187 points are height scans
        d_obs = x[:, :-self.terrain_dim]  # Rest are dynamics (friction, mass, etc.)

        t_lat = self.terrain_branch(t_obs)
        d_lat = self.dynamics_branch(d_obs)

        combined = torch.cat([t_lat, d_lat], dim=-1)
        return self.head(combined)


class RMAAdaptationModule(nn.Module):
    def __init__(self, history_len=30, proprio_dim=48, latent_dim=8):
        """
        1D-CNN Adaptation Module: Infers latent z from proprioceptive history.
        Architecture follows the RMA paper (1D Conv layers).
        """
        super().__init__()

        self.history_len = history_len
        self.proprio_dim = proprio_dim

        self.cnn = nn.Sequential(
            nn.Conv1d(proprio_dim, 32, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten(),
        )

        # Compute output size for the linear layer dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, proprio_dim, history_len)
            cnn_out_dim = self.cnn(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim, 32),
            nn.ELU(),
            nn.Linear(32, latent_dim),
            nn.Tanh(),
        )

        # Initialize
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, history):
        # history: [Batch, HistoryLen, ProprioDim]
        # Conv1d expects [Batch, ProprioDim, HistoryLen]
        x = history.transpose(1, 2)
        features = self.cnn(x)
        return self.head(features)


class RMAActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        # Filter out rma-specific kwargs before passing to super().__init__
        # to avoid "unexpected arguments" warnings from rsl_rl
        rsl_rl_kwargs = {k: v for k, v in kwargs.items() if k not in ["num_actor_obs", "num_critic_obs", "pretrained_path"]}
        super().__init__(obs, obs_groups, num_actions, **rsl_rl_kwargs)

        # Save device reference from observations
        self.device = obs["policy"].device

        # --- QUALIFICATION / REPRODUCTION MODE ---
        # "normal": compute z from privileged info (Matches Stage B/D training)
        # "zero":   force z to 0.0 (baseline check)
        # "shuffled": randomize z across batch (causality check)
        self.latent_mode = "normal"
        # --- TEACHER LATENT ENCODER (Privileged -> Latent) ---
        self.proprio_dim = 48
        self.latent_dim = 8
        self.privileged_dim = 204

        self.encoder = RMALatentEncoder(input_dim=self.privileged_dim, latent_dim=self.latent_dim).to(self.device)

        # 🔴 FORCE ACTOR INPUT to (48 + 8) = 56 for checkpoint compatibility
        # RSL-RL usually builds the MLP based on this value
        self.num_actor_obs = self.proprio_dim + self.latent_dim

        # 🔴 RECONSTRUCT OFFSET TABLE for flat tensor support
        self.obs_offsets = {}
        self.obs_dims = {}
        curr_offset = 0
        for group in obs_groups.keys():
            dim = sum([obs[g].shape[-1] for g in obs_groups[group]])
            self.obs_offsets[group] = curr_offset
            self.obs_dims[group] = dim
            curr_offset += dim

        # Override the actor MLP if it was created with wrong size (48)
        # We need 56 for Stage B/D checkpoints
        if hasattr(self, "actor") and self.actor[0].in_features != self.num_actor_obs:
            print(f"  [RMA] Rebuilding Actor MLP (48 -> {self.num_actor_obs})")
            from rsl_rl.networks.mlp import MLP
            self.actor = MLP(
                input_dim=self.num_actor_obs,
                output_dim=num_actions,
                hidden_dims=kwargs.get("actor_hidden_dims", [512, 256, 128]),
                activation=kwargs.get("activation", "elu"),
            ).to(self.device)

        # 🔴 BOOTSTRAP: Load pre-trained weights if provided
        pretrained_path = kwargs.get("pretrained_path", None)
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)

    def _get_group_obs(self, obs, group_name):
        """Helper to get a group obs from either a dict, TensorDict, or a flat tensor."""
        # 1. If it's a dict/TensorDict, extract the requested group
        if hasattr(obs, "keys"): # Catches both dict and TensorDict
             # In play.py, obs might be JUST the 'policy' group tensor
             # Try to get the key directly first
             if group_name in obs:
                 v = obs[group_name]
                 # 🔴 FIX: If it's a TensorDict, convert to raw Tensor for feature-dim slicing
                 if torch.is_tensor(v):
                     return v
                 # Handle TensorDict or nested dict: collect all leaves and cat them
                 if hasattr(v, "values"):
                     return torch.cat([val for val in v.values() if torch.is_tensor(val)], dim=-1)
                 try:
                    return torch.as_tensor(v) # Last ditch attempt
                 except:
                    return v # Fallback

             # Otherwise, reconstruct from subgroups if all keys exist
             subgroups = self.obs_groups.get(group_name, [])
             if all(k in obs for k in subgroups):
                 parts = []
                 for k in subgroups:
                     if k == group_name: continue
                     part = obs[k]
                     # 🔴 FIX: Ensure part is a raw tensor
                     if not torch.is_tensor(part):
                         if hasattr(part, "values"):
                             part = torch.cat([val for val in part.values() if torch.is_tensor(val)], dim=-1)
                         else:
                             try: part = torch.as_tensor(part)
                             except: pass
                     parts.append(part)
                 if parts:
                    return torch.cat(parts, dim=-1)

        # 2. If it's a flat tensor, slice it using pre-calculated offsets
        if torch.is_tensor(obs):
            # If the tensor size matches the requested group exactly, it's already sliced
            if obs.shape[-1] == self.obs_dims.get(group_name, -1):
                return obs

            # Otherwise, slice from a full-state tensor
            start = self.obs_offsets.get(group_name, 0)
            end = start + self.obs_dims.get(group_name, 0)
            if end <= obs.shape[-1]:
                return obs[:, start:end]

        # Fallback: hope it's what we need
        return obs

    def get_actor_obs(self, obs_dict):
        # Extract base policy observation
        base_obs = self._get_group_obs(obs_dict, "policy")

        # Compute z dynamically inside the policy forward graph
        privileged_obs = self._get_group_obs(obs_dict, "critic")
        # In our asymmetric setup, critic = policy + privileged_info
        # We only want the privileged_info part (last 204 dims).
        # 🔴 FIX: Robust slicing to handle both [Batch, Dim] and [Dim] shapes.
        if privileged_obs.dim() == 1:
             privileged_info = privileged_obs[-204:]
        else:
             privileged_info = privileged_obs[:, -204:]

        # Robustness: Clean up NaNs/Infs in privileged observations
        if torch.isnan(privileged_info).any() or torch.isinf(privileged_info).any():
            privileged_info = torch.nan_to_num(privileged_info, nan=0.0, posinf=0.0, neginf=0.0)

        # Add a tiny bit of noise if needed
        # if self.training:
        #     privileged_obs = privileged_obs + torch.randn_like(privileged_obs) * 0.05

        # The encoder only sees hidden privileged context, never duplicated proprioception.
        z = self.encoder(privileged_info)

        # --- LATENT INSTRUMENTATION ---
        if not hasattr(self, '_fwd_step'):
            self._fwd_step = 0
        self._fwd_step += 1

        # Print stats roughly every 100 env steps (few times per PPO iteration)
        if self._fwd_step % 100 == 0:
            z_norm = torch.norm(z, dim=-1).mean().item()
            z_std = z.std(dim=0).mean().item()
            print(f"  [Latent Probe] ||z||: {z_norm:.4f} | std(z) across batch: {z_std:.4f}")
        # ------------------------------

        # --- QUALIFICATION LOGIC ---
        if self.latent_mode == "zero":
            z = torch.zeros_like(z)
        elif self.latent_mode == "shuffled":
            z = z[torch.randperm(z.size(0))]
        # ----------------------------

        # Actor receives proprio + latent
        full_obs = torch.cat([base_obs, z], dim=-1)

        # Final safety check before passing to Actor
        if torch.isnan(full_obs).any():
            full_obs = torch.nan_to_num(full_obs, nan=0.0)

        return full_obs


    def load_pretrained_weights(self, path):
        """Load compatible weights from a blind policy checkpoint.

        The flat expert may use wider hidden layers than the rough teacher. In that case,
        only overlapping tensor slices are copied, and incompatible layers are skipped.
        """
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint.get("model_state_dict", checkpoint)
        own_state = self.state_dict()

        for name, param in model_state.items():
            if name not in own_state:
                print(f"  [Bootstrap] Warning: Layer {name} from checkpoint not found in current model.")
                continue
            if "std" in name:
                print(f"  [Bootstrap] Skipping {name} to preserve high target exploration noise.")
                continue

            target = own_state[name]
            if target.ndim == 2 and param.ndim == 2:
                rows = min(target.shape[0], param.shape[0])
                cols = min(target.shape[1], param.shape[1])
                with torch.no_grad():
                    target[:rows, :cols].copy_(param[:rows, :cols])
                print(
                    f"  [Bootstrap] Partially copied {name}: "
                    f"{tuple(param.shape)} -> {tuple(target.shape)} using ({rows}, {cols})"
                )
            elif target.ndim == 1 and param.ndim == 1:
                size = min(target.shape[0], param.shape[0])
                with torch.no_grad():
                    target[:size].copy_(param[:size])
                print(
                    f"  [Bootstrap] Partially copied {name}: "
                    f"{tuple(param.shape)} -> {tuple(target.shape)} using ({size},)"
                )
            else:
                try:
                    target.copy_(param)
                    print(f"  [Bootstrap] Copied {name}: {tuple(param.shape)}")
                except RuntimeError as e:
                    print(f"  [Bootstrap] Warning: Skipping layer {name} due to mismatch: {e}")

        self.load_state_dict(own_state)
        print(f"  [Bootstrap] Loaded pre-trained weights from: {path}")


    # Critic does not need to be overridden. It inherits exactly what was defined in the superclass,
    # which uses self.obs_groups["critic"] safely.


class RMAStudentActorCritic(RMAActorCritic):
    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        """
        Student Architecture: Replaces privileged encoder with Adaptation Module.
        """
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        # History is now inside the 'policy' group: [Batch, ProprioDim + HistoryLen * ProprioDim]
        self.proprio_dim = 48
        policy_group_dim = self.obs_dims["policy"]
        self.history_len = (policy_group_dim - self.proprio_dim) // self.proprio_dim

        latent_dim = 8

        # 🔴 CRITICAL: FORCE ACTOR DIMS to [Proprio + Latent]
        # We override the base class calculation so the MLP is built with 56 inputs
        self.num_actor_obs = self.proprio_dim + latent_dim
        self.adaptation_module = RMAAdaptationModule(
            history_len=self.history_len,
            proprio_dim=self.proprio_dim,
            latent_dim=latent_dim
        ).to(self.device)

        # The teacher's actor/critic and encoder are maintained for distillation
        self.teacher_encoder = self.encoder # Reuse base teacher encoder for supervised loss
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False # Freeze teacher encoder

        self.is_auditing = False # 🟢 TOGGLE THIS TO FALSE FOR 'TURBO' SPEED

    def get_actor_obs(self, obs_dict):
        # 1. Base proprioception (The first 48 dims of the policy group)
        policy_obs = self._get_group_obs(obs_dict, "policy")

        # 🔴 ROBUSTNESS: If the 'policy' group is only 48-dim, but 'history' is available separately,
        # manually join them to ensure the adaptation module receives data.
        if policy_obs.shape[-1] == self.proprio_dim:
             if isinstance(obs_dict, dict) and "history" in obs_dict:
                  history_part = self._get_group_obs(obs_dict, "history")
                  policy_obs = torch.cat([policy_obs, history_part], dim=-1)

        base_obs = policy_obs[:, :self.proprio_dim]

        # 2. Adaptation: Predict z from history (The rest of the policy group)
        history_flat = policy_obs[:, self.proprio_dim:]

        # 🔴 ONE-SHOT BOOT DIAGNOSTIC
        if not hasattr(self, '_boot_diag_done'):
            self._boot_diag_done = True
            print(f"  [BOOT] policy_obs.shape={policy_obs.shape} | history_flat.shape={history_flat.shape} | history_len={self.history_len}")

        # If history_flat is unexpectedly empty, return zero latent to avoid cat() batch mismatch (1 vs 0)
        if history_flat.numel() == 0:
             z_predicted = torch.zeros((policy_obs.shape[0], 8), device=self.device)
        else:
             history = history_flat.view(-1, self.history_len, self.proprio_dim)
             z_predicted = self.adaptation_module(history)

             # --- VIGOROUS AUDIT ---
             if self.is_auditing:
                  if not hasattr(self, '_audit_step'): self._audit_step = 0
                  self._audit_step += 1
                  if self._audit_step % 1000 == 0:
                       p_min, p_max = base_obs.min().item(), base_obs.max().item()
                       h_min, h_max = history_flat.min().item(), history_flat.max().item()
                       h_mean = history_flat.abs().mean().item()
                       z_mean = z_predicted.mean().item()
                       print(f"  [Audit] Proprio [{p_min:.2f}, {p_max:.2f}] | History Mag: {h_mean:.4f} | Z_mean: {z_mean:.3f}")
                       print(f"  [Audit] Env0 v_xy: ({base_obs[0,0]:.2f}, {base_obs[0,1]:.2f}) m/s")

        # 3. Cache prediction for distillation loss (used by PPO)
        self.last_predicted_z = z_predicted

        # Student policy receives proprio + predicted latent
        return torch.cat([base_obs, z_predicted], dim=-1)