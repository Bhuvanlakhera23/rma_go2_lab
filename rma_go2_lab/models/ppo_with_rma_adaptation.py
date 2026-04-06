# rma_go2_lab/models/ppo_with_rma_adaptation.py
from __future__ import annotations
import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

class PPOWithRMAAdaptation(PPO):
    def __init__(
        self,
        policy,
        *args,
        adaptation_loss_coef: float = 10.0, # High weight to force latent tracking
        **kwargs,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        self.adaptation_loss_coef = float(adaptation_loss_coef)
        self.imitation_loss_fn = nn.MSELoss()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_adaptation_loss = 0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            
            # 1. Standard PPO forward
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # 2. Adaptation Distillation Loss
            # Student AC caches the last_predicted_z during self.policy.act()
            z_predicted = self.policy.last_predicted_z
            with torch.no_grad():
                # Teacher encoder uses privileged info (terrain + dynamics)
                z_teacher = self.policy.teacher_encoder(obs_batch["privileged"])
            
            adaptation_loss = self.imitation_loss_fn(z_predicted, z_teacher)

            # 3. Standard KL/Surrogate math
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # TOTAL LOSS: Surrogate + Value + Entropy + 🟢 ADAPTATION
            loss = (
                surrogate_loss 
                + self.value_loss_coef * value_loss 
                - self.entropy_coef * entropy_batch.mean()
                + self.adaptation_loss_coef * adaptation_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_adaptation_loss += adaptation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "adaptation_loss": mean_adaptation_loss / num_updates,
        }
