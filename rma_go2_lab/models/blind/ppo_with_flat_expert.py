from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO

from rma_go2_lab.models.blind.frozen_flat_expert import FrozenFlatExpert


class BlindPPOWithFlatExpert(PPO):
    """PPO with an optional time-decayed imitation prior from a flat expert."""

    def __init__(
        self,
        policy,
        *args,
        flat_expert_path: str | None = None,
        flat_expert_activation: str = "elu",
        flat_imitation_command_threshold: float = 0.1,
        flat_imitation_coef_stage0: float = 0.3,
        flat_imitation_coef_stage1: float = 0.1,
        flat_imitation_stage0_end: int = 300,
        flat_imitation_stage1_end: int = 800,
        **kwargs,
    ) -> None:
        super().__init__(policy, *args, **kwargs)
        self.flat_imitation_command_threshold = float(flat_imitation_command_threshold)
        self.flat_imitation_coef_stage0 = float(flat_imitation_coef_stage0)
        self.flat_imitation_coef_stage1 = float(flat_imitation_coef_stage1)
        self.flat_imitation_stage0_end = int(flat_imitation_stage0_end)
        self.flat_imitation_stage1_end = int(flat_imitation_stage1_end)
        self._update_counter = 0
        self.flat_expert = None
        self.imitation_loss_fn = nn.MSELoss(reduction="none")

        if flat_expert_path:
            self.flat_expert = FrozenFlatExpert(
                checkpoint_path=flat_expert_path,
                activation=flat_expert_activation,
                device=self.device,
            ).to(self.device)

    def _current_imitation_coef(self) -> float:
        if self._update_counter < self.flat_imitation_stage0_end:
            return self.flat_imitation_coef_stage0
        if self._update_counter < self.flat_imitation_stage1_end:
            return self.flat_imitation_coef_stage1
        return 0.0

    def _imitation_mask(self, obs_batch) -> torch.Tensor | None:
        if self.flat_expert is None or "policy" not in obs_batch.keys():
            return None
        command = obs_batch["policy"][:, 9:12]
        return (torch.linalg.norm(command, dim=-1) > self.flat_imitation_command_threshold).float()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_imitation_loss = 0
        mean_imitation_active_frac = 0
        mean_imitation_coef = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

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
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            imitation_loss = torch.tensor(0.0, device=self.device)
            active_frac = torch.tensor(0.0, device=self.device)
            imitation_coef = self._current_imitation_coef()
            if self.flat_expert is not None and imitation_coef > 0.0:
                expert_obs_batch = obs_batch[:original_batch_size]
                mask = self._imitation_mask(expert_obs_batch)
                if mask is not None:
                    active_frac = mask.mean()
                    if torch.count_nonzero(mask) > 0:
                        flat_actions = self.flat_expert(expert_obs_batch["policy"]).detach()
                        per_sample = self.imitation_loss_fn(mu_batch, flat_actions).sum(dim=-1)
                        imitation_loss = (per_sample * mask).sum() / (mask.sum() + 1e-6)
                        loss = loss + imitation_coef * imitation_loss

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_imitation_loss += imitation_loss.item()
            mean_imitation_active_frac += active_frac.item()
            mean_imitation_coef += imitation_coef
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_imitation_loss /= num_updates
        mean_imitation_active_frac /= num_updates
        mean_imitation_coef /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()
        self._update_counter += 1

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "flat_imitation": mean_imitation_loss,
            "flat_imitation_active_frac": mean_imitation_active_frac,
            "flat_imitation_coef": mean_imitation_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
