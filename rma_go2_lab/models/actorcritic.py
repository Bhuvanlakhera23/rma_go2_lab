import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from rsl_rl.networks import MLP
from rsl_rl.networks import EmpiricalNormalization


class RMAActorCritic(nn.Module):

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.5,
        state_dependent_std=True,
        **kwargs,
    ):
        super().__init__()
        self.debug_step = 0
        self.state_dependent_std = state_dependent_std

        # --------------------------------------------------
        # Observation Dimensions
        # --------------------------------------------------
        self.policy_obs_keys = obs_groups["policy"]

        num_policy_obs = sum(obs[g].shape[-1] for g in self.policy_obs_keys)
        num_privileged = obs["privileged"].shape[-1]

        # --------------------------------------------------
        # Actor Network
        # --------------------------------------------------
        actor_input_dim = num_policy_obs

        # feature extractor
        self.actor_body = MLP(
            actor_input_dim,
            actor_hidden_dims[-1],
            actor_hidden_dims[:-1],
            activation,
        )

        self.mean_head = nn.Linear(actor_hidden_dims[-1], num_actions)

        if state_dependent_std:
            self.log_std_head = nn.Linear(actor_hidden_dims[-1], num_actions)
        else:
            self.log_std = nn.Parameter(
                init_noise_std * torch.ones(num_actions)
            )

        # --------------------------------------------------
        # Critic Network
        # --------------------------------------------------
        critic_input_dim = num_policy_obs + num_privileged

        self.critic = MLP(
            critic_input_dim,
            1,
            critic_hidden_dims,
            activation,
        )

        # --------------------------------------------------
        # Normalization
        # --------------------------------------------------
        self.actor_obs_normalizer = (
            EmpiricalNormalization(actor_input_dim)
            if actor_obs_normalization
            else nn.Identity()
        )

        self.critic_obs_normalizer = (
            EmpiricalNormalization(critic_input_dim)
            if critic_obs_normalization
            else nn.Identity()
        )

        self.distribution = None
        Normal.set_default_validate_args(False)

        print("---- RMA NETWORK INFO ----")
        print("Policy obs dim:", num_policy_obs)
        print("Privileged dim:", num_privileged)
        print("Actor input dim:", actor_input_dim)
        print("Critic input dim:", critic_input_dim)
        print("---------------------------")

    # ======================================================
    # Observation Processing
    # ======================================================

    def get_actor_obs(self, obs: TensorDict):
        return torch.cat([obs[g] for g in self.policy_obs_keys], dim=-1)

    def get_critic_obs(self, obs: TensorDict):
        policy_obs = torch.cat([obs[g] for g in self.policy_obs_keys], dim=-1)
        privileged = obs["privileged"]
        return torch.cat([policy_obs, privileged], dim=-1)

    # ======================================================
    # Distribution
    # ======================================================

    def _update_distribution(self, actor_input):
        
        features = self.actor_body(actor_input)
        mean = self.mean_head(features)

        if self.state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -5, 2)
            std = torch.exp(log_std)
        else:
            std = torch.exp(self.log_std).expand_as(mean)

        self.action_mean = mean
        self.action_std = std
        self.distribution = Normal(mean, std)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ======================================================
    # Acting
    # ======================================================

    def act(self, obs: TensorDict, **kwargs):

        actor_input = self.get_actor_obs(obs)
        actor_input = self.actor_obs_normalizer(actor_input)

        self._update_distribution(actor_input)

        self.debug_step += 1
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict):
        actor_input = self.get_actor_obs(obs)
        actor_input = self.actor_obs_normalizer(actor_input)
        features = self.actor_body(actor_input)
        return self.mean_head(features)

    def evaluate(self, obs: TensorDict, **kwargs):
        critic_input = self.get_critic_obs(obs)
        critic_input = self.critic_obs_normalizer(critic_input)
        return self.critic(critic_input)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ======================================================
    # Normalization Update
    # ======================================================

    def update_normalization(self, obs: TensorDict):

        actor_input = self.get_actor_obs(obs)
        critic_input = self.get_critic_obs(obs)

        if isinstance(self.actor_obs_normalizer, EmpiricalNormalization):
            self.actor_obs_normalizer.update(actor_input)

        if isinstance(self.critic_obs_normalizer, EmpiricalNormalization):
            self.critic_obs_normalizer.update(critic_input)
    
    def reset(self, dones=None):
        # Non-recurrent policy — nothing to reset
        return