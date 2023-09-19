import pickle
from os import path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger
from sc2.data import Result
from sc2.position import Point2
from sc2.units import Units
from torch import Tensor, nn, optim
from torch.nn.utils import clip_grad_norm_

from bot.consts import ConfigSettings
from bot.squad_agent.agents.base_agent import BaseAgent
from bot.squad_agent.architecture.dqn_rainbow.model import Model
from bot.squad_agent.architecture.dqn_rainbow.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from bot.squad_agent.features import Features
from bot.squad_agent.utils import load_checkpoint, save_checkpoint


class DQNRainbowAgent(BaseAgent):
    def __init__(self, ai, config: Dict) -> None:
        super().__init__(ai, config)
        state_dir: str = self.config[ConfigSettings.SQUAD_AGENT][
            ConfigSettings.STATE_DIRECTORY
        ]
        self.save_replay_buffer_path = f"{self.DATA_DIR}/{state_dir}/replay_buffer.pkl"
        self.PICKLE_REPLAY_BUFFER_PATH: str = path.join(
            self.DATA_DIR,
            "replay_buffer.pkl",
        )
        self.PICKLE_REPLAY_BUFFER_N_PATH: str = path.join(
            self.DATA_DIR,
            "replay_buffer_n.pkl",
        )

        self._loss: float = 0.0
        self.last_action: int = 0
        self.prev_episode_reward: float = (
            5.0 if self.all_episode_data[-1]["Result"] == 2 else -5.0
        )
        dqn_settings: dict = self.config[ConfigSettings.SQUAD_AGENT][ConfigSettings.DQN]
        grid = self.ai.mediator.get_ground_grid
        obs_dim = 292
        action_dim = 5
        self.batch_size = dqn_settings[ConfigSettings.BATCH_SIZE]
        self.target_update = dqn_settings[ConfigSettings.TARGET_UPDATE]
        self.gamma = dqn_settings[ConfigSettings.GAMMA]

        self.features: Features = Features(ai, config, 256, self.device)

        self.beta = dqn_settings[ConfigSettings.BETA]
        self.prior_eps = dqn_settings[ConfigSettings.PRIOR_EPS]
        if path.isfile(self.PICKLE_REPLAY_BUFFER_PATH):
            with open(self.PICKLE_REPLAY_BUFFER_PATH, "rb") as f:
                self.memory = pickle.load(f)
                self.memory.done_buf[self.memory.ptr] = True
                self.memory.rews_buf[self.memory.ptr] += self.prev_episode_reward
                logger.info("Loaded existing prioritized replay buffer")
        else:
            self.memory = PrioritizedReplayBuffer(
                self.device,
                obs_dim,
                dqn_settings[ConfigSettings.MEMORY_SIZE],
                self.batch_size,
                alpha=dqn_settings[ConfigSettings.ALPHA],
            )
        self.n_step = 3
        # memory for N-step Learning
        self.use_n_step = self.n_step > 1
        if self.use_n_step:
            if path.isfile(self.PICKLE_REPLAY_BUFFER_N_PATH):
                with open(self.PICKLE_REPLAY_BUFFER_N_PATH, "rb") as f:
                    self.memory_n = pickle.load(f)
                    self.memory_n.done_buf[self.memory_n.ptr] = True
                    self.memory_n.rews_buf[self.memory.ptr] += self.prev_episode_reward
                    logger.info("Loaded existing replay buffer")
            else:
                self.memory_n = ReplayBuffer(
                    self.device,
                    obs_dim,
                    dqn_settings[ConfigSettings.MEMORY_SIZE],
                    self.batch_size,
                    n_step=self.n_step,
                    gamma=dqn_settings[ConfigSettings.GAMMA],
                )

        # Categorical DQN parameters
        self.v_min = dqn_settings[ConfigSettings.V_MIN]
        self.v_max = dqn_settings[ConfigSettings.V_MAX]
        self.atom_size = dqn_settings[ConfigSettings.ATOM_SIZE]
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
            self.device
        )
        self.epoch: int = 0

        self.model: Model = Model(
            grid,
            self.ai.game_info.map_size[1],
            self.ai.game_info.map_size[0],
            obs_dim,
            action_dim,
            self.support,
            self.device,
            self.atom_size,
        ).to(self.device)

        # optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=dqn_settings[ConfigSettings.LEARNING_RATE]
        )
        self.update_steps: int = 0

        if path.isfile(self.CHECKPOINT_PATH):
            (
                self.model,
                self.optimizer,
                self.epoch,
                self.beta,
                self.update_steps,
            ) = load_checkpoint(
                self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
            )
            logger.info(f"Loaded existing model at {self.CHECKPOINT_PATH}")
            logger.info(f"Epoch: {self.epoch}")
        else:
            save_checkpoint(
                self.CHECKPOINT_PATH,
                self.epoch,
                self.model,
                self.optimizer,
                self.beta,
                self.update_steps,
            )

        if self.training_active:
            self.model.train()
            self.model.dqn_target.eval()
        else:
            self.model.eval()

        # transition to store in memory
        self.transition = []

        # need to store the previous state between frames
        self.state: Tensor = torch.zeros((1, 292))

    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
        attack_target: Point2,
        rally_point: Point2,
    ) -> int:
        super(DQNRainbowAgent, self).choose_action(
            squads,
            pos_of_squad,
            all_close_enemy,
            squad_units,
            attack_target,
            rally_point,
        )
        self.cumulative_reward += self.reward
        obs = self.features.transform_obs(
            self.pathing.ground_grid,
            self.pathing.effects_grid,
            pos_of_squad,
            attack_target,
            rally_point,
        )
        spatial, entity, scalar, locations = obs
        locations = locations.to(self.device)
        spatial = spatial.to(self.device)
        entity = entity.to(self.device)
        entity = nn.functional.normalize(entity)
        hidden, processed_spatial = self.model.encoder(
            spatial, entity, scalar, locations, True
        )
        action: int = self.select_action(hidden)
        if self.training_active:
            transition = [self.state, int(action)]
            self.last_action = int(action)
            self.train(hidden, transition)

        self.squad_reward = 0.0

        return action

    def select_action(self, state: torch.Tensor) -> int:
        selected_action = self.model.dqn(state).argmax()
        selected_action = selected_action.detach().cpu().numpy().item()

        self.action_distribution[selected_action] += 1
        return selected_action

    def train(self, hidden: torch.Tensor, transition: list, done: float = 0.0) -> None:
        # store transition to the next state
        transition += [self.reward, hidden, done]

        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*transition)
        # 1-step transition
        else:
            one_step_transition = transition

        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)

        self.state = hidden

        # PER: increase beta
        fraction = min(self.epoch / 10000000000, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        # if training is ready
        if len(self.memory) >= self.batch_size:
            (
                self.model,
                self.optimizer,
                self.epoch,
                beta,
                self.update_steps,
            ) = load_checkpoint(
                self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
            )
            self.model.dqn_target.eval()
            _loss = self.update_model()
            self._loss = _loss
            _beta = self.beta if self.beta > beta else beta

            self.update_steps += 1
            # if hard update is needed
            if self.update_steps % self.target_update == 0:
                logger.info("Updating DQN target model")
                self.update_steps = 0
                self.model.dqn_target.load_state_dict(self.model.dqn.state_dict())

            save_checkpoint(
                self.CHECKPOINT_PATH,
                self.epoch,
                self.model,
                self.optimizer,
                _beta,
                self.update_steps,
            )
            self.epoch += 1

            self.writer.add_scalar("hyperparameter/beta", self.beta, self.epoch)
            self.writer.add_scalar("losses/loss", _loss, self.epoch)

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)

        if self.use_n_step:
            gamma = self.gamma**self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.dqn.parameters(), 10.0)
        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.model.dqn.reset_noise()
        self.model.dqn_target.reset_noise()

        return loss.item()

    def on_episode_end(self, result: Result, update_model: bool = False) -> None:
        logger.info("On episode end called")
        _reward: float = 5.0 if result == Result.Victory else -5.0
        with open(f"{self.DATA_DIR}/replay_buffer.pkl", "wb") as f:
            pickle.dump(self.memory, f)

        with open(f"{self.DATA_DIR}/replay_buffer_n.pkl", "wb") as f:
            pickle.dump(self.memory_n, f)

        self.store_episode_data(
            result=result,
            steps=self.epoch,
            reward=self.cumulative_reward + _reward,
            action_distribution=self.action_distribution,
            loss=self._loss,
        )
        self.squad_reward += _reward
        transition = [self.state, self.last_action]
        if update_model:
            self.train(self.state, transition, done=1.0)

        self.writer.add_scalar("rewards/cum_reward", self.cumulative_reward)
        logger.info("Trained model on episode end")

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float
    ) -> torch.Tensor:
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].reshape(-1, 1)
        done = samples["done"].reshape(-1, 1)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.model.dqn(next_state).argmax(1)
            next_dist = self.model.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            _l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (_l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - _l.float())).view(-1)
            )

        dist = self.model.dqn.dist(state)
        indices = torch.Tensor(range(self.batch_size)).to(torch.int64)
        log_p = torch.log(dist[indices, action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
