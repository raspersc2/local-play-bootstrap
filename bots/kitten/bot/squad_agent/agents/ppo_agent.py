"""
The offline agent goal is to collect state, actions and rewards and store them to disk
(Note: Fine to use this agent to test a trained model by setting InferenceMode: True)
RL Training (back propagation) should then be carried out
via a separate process / script after the game is complete
"""
from os import path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from sc2.data import Result
from sc2.position import Point2
from sc2.units import Units
from torch import Tensor, nn, optim

from bot.consts import SQUAD_ACTIONS, ConfigSettings
from bot.squad_agent.agents.base_agent import BaseAgent
from bot.squad_agent.architecture.ppo.actor_critic import ActorCritic
from bot.squad_agent.features import Features
from bot.squad_agent.utils import load_checkpoint, save_checkpoint

NUM_ENVS: int = 1
SPATIAL_SHAPE: tuple[int, int, int, int] = (1, 38, 120, 120)
ENTITY_SHAPE: tuple[int, int, int] = (1, 256, 406)
SCALAR_SHAPE: tuple[int, int] = (1, 8)


class PPOAgent(BaseAgent):
    __slots__ = (
        "features",
        "model",
        "optimizer",
        "initial_lstm_state",
        "current_lstm_state",
        "entities",
        "spatials",
        "scalars",
        "actions",
        "locations",
        "logprobs",
        "rewards",
        "dones",
        "values",
        "current_rollout_step",
        "game_id",
        "data_chunk",
        "num_rollout_steps",
        "clip_coefficient",
        "entropy_coefficient",
        "batch_size",
        "gae_lambda",
        "gamma",
        "max_grad_norm",
        "update_policy_epochs",
        "vf_coefficient",
        "value_loss",
        "policy_loss",
    )

    def __init__(self, ai, config: Dict):
        # we will use the aiarena docker to play multiple simultaneous games
        # to collect state, action, rewards etc.
        # so use "cpu" here and
        # the separate training script should use "cuda" if available
        super().__init__(ai, config, "cpu")

        self.features: Features = Features(ai, config, 256, self.device)

        ppo_settings: dict = self.config[ConfigSettings.SQUAD_AGENT][ConfigSettings.PPO]
        self.num_rollout_steps: int = ppo_settings[ConfigSettings.NUM_ROLLOUT_STEPS]

        grid = self.ai.mediator.get_ground_grid

        self.model = ActorCritic(
            len(SQUAD_ACTIONS),
            self.device,
            grid,
            self.ai.game_info.map_size[1],
            self.ai.game_info.map_size[0],
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        if path.isfile(self.CHECKPOINT_PATH):
            self.model, self.optimizer, self.epoch = load_checkpoint(
                self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
            )
            logger.info(f"Loaded existing model at {self.CHECKPOINT_PATH}")
        # nothing stored on disk yet,
        # there should be something there for the training script later
        else:
            save_checkpoint(
                self.CHECKPOINT_PATH, self.epoch, self.model, self.optimizer
            )

        if self.training_active:
            self.model.train()
        else:
            self.model.eval()

        self.initial_lstm_state = (
            torch.zeros(
                self.model.lstm.num_layers, NUM_ENVS, self.model.lstm.hidden_size
            ).to(self.device),
            torch.zeros(
                self.model.lstm.num_layers, NUM_ENVS, self.model.lstm.hidden_size
            ).to(self.device),
        )

        self.current_lstm_state = self.initial_lstm_state
        num_rollout_steps = self.num_rollout_steps
        self.entities = torch.zeros((num_rollout_steps,) + ENTITY_SHAPE).to(self.device)
        self.spatials = torch.zeros((num_rollout_steps,) + SPATIAL_SHAPE).to(
            self.device
        )
        self.scalars = torch.zeros((num_rollout_steps,) + SCALAR_SHAPE).to(self.device)
        self.actions = torch.zeros((num_rollout_steps,) + (1,)).to(self.device)
        self.locations = torch.zeros((num_rollout_steps,) + (1, 256, 2)).to(self.device)
        self.logprobs = torch.zeros((num_rollout_steps, NUM_ENVS)).to(self.device)
        self.rewards = torch.zeros((num_rollout_steps, NUM_ENVS)).to(self.device)
        self.dones = torch.zeros((num_rollout_steps, NUM_ENVS)).to(self.device)
        self.values = torch.zeros((num_rollout_steps, NUM_ENVS)).to(self.device)
        self.current_rollout_step: int = 0

        self.data_chunk: int = 0

        self.clip_coefficient: float = ppo_settings[ConfigSettings.CLIP_COEFFICIENT]
        self.entropy_coefficient: float = ppo_settings[
            ConfigSettings.ENTROPY_COEFFICIENT
        ]
        self.batch_size: int = ppo_settings[ConfigSettings.BATCH_SIZE]
        self.gae_lambda: float = ppo_settings[ConfigSettings.GAE_LAMBDA]
        self.gamma: float = ppo_settings[ConfigSettings.GAMMA]
        self.max_grad_norm: float = ppo_settings[ConfigSettings.MAX_GRAD_NORM]
        self.update_policy_epochs: int = ppo_settings[
            ConfigSettings.UPDATE_POLICY_EPOCHS
        ]
        self.vf_coefficient: float = ppo_settings[ConfigSettings.VF_COEFFICIENT]
        self.value_loss = 0.0
        self.policy_loss = 0.0

    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
        attack_target: Point2,
        rally_point: Point2,
    ) -> int:
        super(PPOAgent, self).choose_action(
            squads,
            pos_of_squad,
            all_close_enemy,
            squad_units,
            attack_target,
            rally_point,
        )
        reward: float = self.reward
        obs = self.features.transform_obs(
            self.ai.mediator.get_ground_grid,
            self.ai.mediator.get_ground_avoidance_grid,
            pos_of_squad,
            attack_target,
            rally_point,
        )
        spatial, entity, scalar, locations = obs
        locations = locations.to(self.device)
        spatial = spatial.to(self.device)
        entity = entity.to(self.device)
        entity = nn.functional.normalize(entity)

        self.cumulative_reward += reward

        with torch.no_grad():
            (
                action,
                logprob,
                _,
                value,
                self.current_lstm_state,
                processed_spatial,
            ) = self.model.get_action_and_value(
                spatial,
                entity,
                scalar,
                locations,
                self.current_lstm_state,
                self.dones,
            )
            if self.visualize_spatial_features:
                self._plot_spatial_features(processed_spatial[0])
            self.action_distribution[action] += 1

            step: int = self.current_rollout_step

            if self.training_active:
                if self.current_rollout_step < self.num_rollout_steps:
                    self.current_rollout_step += 1
                    self.entities[step] = entity
                    self.scalars[step] = scalar
                    self.spatials[step] = processed_spatial
                    self.locations[step] = locations
                    self.actions[step] = action
                    self.logprobs[step] = logprob
                    self.rewards[step] = self.reward
                    self.values[step] = value
                    self.squad_reward = 0.0
                else:
                    # load the up-to-date model,
                    # or we just be overwriting other processes
                    self.model, self.optimizer, self.epoch = load_checkpoint(
                        self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
                    )
                    self.current_rollout_step = 0
                    logger.info("Performing back propagation")
                    self._back_propagation()
                    logger.info("Storing updated model")
                    save_checkpoint(
                        self.CHECKPOINT_PATH, self.epoch, self.model, self.optimizer
                    )

            return action.item()

    def _back_propagation(self) -> None:
        with torch.no_grad():
            actions: Tensor = self.actions
            logprobs: Tensor = self.logprobs
            spatials: Tensor = self.spatials
            entities: Tensor = self.entities
            scalars: Tensor = self.scalars
            locations: Tensor = self.locations
            dones: Tensor = self.dones
            rewards: Tensor = self.rewards
            values: Tensor = self.values

            next_value: float = self.model.get_value(
                spatials[-1],
                entities[-1],
                scalars[-1],
                locations[-1],
                self.current_lstm_state,
                dones[-1],
                False,
            ).reshape(1, -1)

            advantages: Tensor = torch.zeros_like(rewards).to(self.device)
            last_gaelam: float = 0.0
            rollout_steps: int = self.num_rollout_steps
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_non_terminal = 1.0 - 0
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - 0
                    next_values = values[t + 1]
                delta: Tensor = (
                    rewards[t]
                    + self.gamma * next_values * next_non_terminal
                    - values[t]
                )
                advantages[t] = last_gaelam = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gaelam
                )
            returns = advantages + values

            # flatten the batch
            b_entities: Tensor = entities.reshape((-1,) + ENTITY_SHAPE)
            b_entities = torch.squeeze(b_entities)
            b_scalars: Tensor = scalars.reshape((-1,) + SCALAR_SHAPE)
            b_scalars = torch.squeeze(b_scalars)
            b_spatials: Tensor = spatials.reshape((-1,) + SPATIAL_SHAPE)
            b_spatials = torch.squeeze(b_spatials)
            b_locations: Tensor = locations.reshape((-1,) + (256, 2))
            b_locations = torch.squeeze(b_locations)
            b_logprobs: Tensor = logprobs.reshape(-1)
            b_actions: Tensor = actions.reshape(-1)
            b_dones: Tensor = dones.reshape(-1)
            b_advantages: Tensor = advantages.reshape(-1)
            b_returns: Tensor = returns.reshape(-1)
            b_values: Tensor = values.reshape(-1)

            # train the network
            # envsperbatch = NUM_ENVS // NUM_MINIBATCHES
            envinds = np.arange(1)
            flatinds: Tensor = torch.arange(self.batch_size).reshape(rollout_steps, 1)
            # split into 4 minibatches
            mini_batch_ids: list[Tensor] = torch.chunk(flatinds, 4, dim=0)
            current_minibatch: int = 0
            for epoch in range(self.update_policy_epochs):
                if current_minibatch >= 4:
                    current_minibatch = 0
                mbenvinds = envinds
                mb_inds = mini_batch_ids[current_minibatch].squeeze()
                current_minibatch += 1

                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    _,
                    _,
                ) = self.model.get_action_and_value(
                    b_spatials[mb_inds],
                    b_entities[mb_inds],
                    b_scalars[mb_inds],
                    b_locations[mb_inds],
                    (
                        self.current_lstm_state[0][:, mbenvinds],
                        self.current_lstm_state[1][:, mbenvinds],
                    ),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                    process_spatial=False,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                # normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                clip = self.clip_coefficient
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                # clip vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip,
                    clip,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.entropy_coefficient * entropy_loss
                    + v_loss * self.vf_coefficient
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.requires_grad = True
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.optimizer.zero_grad()

                # if args.target_kl is not None:
                if approx_kl > 0.01:
                    break

        self.value_loss = v_loss.item()
        self.policy_loss = pg_loss.item()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        self.writer.add_scalar("rewards/mean_reward", rewards.mean(), self.epoch)
        self.writer.add_scalar("losses/value_loss", self.value_loss, self.epoch)
        self.writer.add_scalar("losses/policy_loss", self.policy_loss, self.epoch)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.epoch)
        self.writer.add_scalar("losses/explained_variance", explained_var, self.epoch)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.epoch)
        self.epoch += 1

    def on_episode_end(self, result: Result) -> None:
        if self.training_active:
            logger.info("On episode end called")
            _reward: float = 5.0 if result == Result.Victory else -5.0
            self.store_episode_data(
                result,
                self.epoch,
                self.cumulative_reward + _reward,
                self.action_distribution,
            )

            current_step: int = self.current_rollout_step
            if current_step == self.num_rollout_steps:
                current_step = self.num_rollout_steps - 1
            self.rewards[current_step] = _reward
            self.dones[current_step] = 1
            # load the up-to-date model, or we just be overwriting other processes
            self.model, self.optimizer, self.epoch = load_checkpoint(
                self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
            )
            self._back_propagation()
            save_checkpoint(
                self.CHECKPOINT_PATH, self.epoch, self.model, self.optimizer
            )
