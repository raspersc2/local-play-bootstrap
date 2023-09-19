"""
The offline agent goal is to collect state, actions and rewards and store them to disk
(Note: Fine to use this agent to test a trained model by setting InferenceMode: True)
RL Training (back propagation) should then be carried out via a separate
process / script after the game is complete
"""
import os
import pickle
import uuid
from os import path
from typing import Dict, List

import torch
from loguru import logger
from sc2.data import Result
from sc2.position import Point2
from sc2.units import Units
from torch import nn, optim

from bot.consts import SQUAD_ACTIONS, ConfigSettings
from bot.squad_agent.agents.base_agent import BaseAgent
from bot.squad_agent.architecture.ppo.actor_critic import ActorCritic
from bot.squad_agent.features import Features
from bot.squad_agent.utils import load_checkpoint, save_checkpoint

NUM_ENVS: int = 1
SPATIAL_SHAPE: tuple[int, int, int, int] = (1, 38, 120, 120)
ENTITY_SHAPE: tuple[int, int, int] = (1, 256, 408)
SCALAR_SHAPE: tuple[int, int] = (1, 8)


class OfflineAgent(BaseAgent):
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
        "save_tensors_path",
        "data_chunk",
        "num_rollout_steps",
    )

    def __init__(self, ai, config: Dict):
        # we will use the aiarena docker to play multiple simultaneous
        # games to collect state, action, rewards etc.
        # so use "cpu" here and the separate training script
        # should use "cuda" if available
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
            self.model, self.optimizer, self.epoch, _, _ = load_checkpoint(
                self.CHECKPOINT_PATH, self.model, self.optimizer, self.device
            )
            logger.info(f"Loaded existing model at {self.CHECKPOINT_PATH}")
        # nothing stored on disk yet,
        # there should be something there for the training script later
        else:
            save_checkpoint(
                self.CHECKPOINT_PATH, self.epoch, self.model, self.optimizer
            )

        # we are not ever training the model here, ensure it stays on evaluation
        # TODO: Probably some optimizations we could do here
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

        # location where to save state, actions, rewards for offline training
        self.game_id: str = uuid.uuid4().hex
        state_dir: str = self.config[ConfigSettings.SQUAD_AGENT][
            ConfigSettings.STATE_DIRECTORY
        ]
        self.save_tensors_path = f"{self.DATA_DIR}/{state_dir}/{self.game_id}/"
        os.makedirs(self.save_tensors_path)

        self.data_chunk: int = 0

    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
        attack_target: Point2,
        rally_point: Point2,
    ) -> int:
        super(OfflineAgent, self).choose_action(
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
                self.entities[step] = entity
                self.scalars[step] = scalar
                self.spatials[step] = processed_spatial
                self.locations[step] = locations
                self.actions[step] = action
                self.logprobs[step] = logprob
                self.rewards[step] = self.reward
                self.values[step] = value
                self.current_rollout_step += 1

                if self.current_rollout_step >= self.num_rollout_steps:
                    self._save_tensors()
                    self.current_rollout_step = 0

            self.squad_reward = 0.0

            return action.item()

    def on_episode_end(self, result: Result) -> None:
        if self.training_active:
            logger.info("On episode end called")
            _reward = 5.0 if result == Result.Victory else -5.0
            if sum(self.action_distribution) > 0:
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
                self.dones[current_step] = 1.0
                self._save_tensors()

    def _save_tensors(self) -> None:

        file_name = f"{self.save_tensors_path}{self.data_chunk}_tensors.pt"
        with open(file_name, "wb") as f:
            try:
                torch.save(
                    {
                        "entities": self.entities,
                        "scalars": self.scalars,
                        "spatials": self.spatials,
                        "locations": self.locations,
                        "actions": self.actions,
                        "logprobs": self.logprobs,
                        "rewards": self.rewards,
                        "dones": self.dones,
                        "values": self.values,
                    },
                    f,
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )
            finally:
                f.flush()
                os.fsync(f.fileno())
        self.data_chunk += 1
