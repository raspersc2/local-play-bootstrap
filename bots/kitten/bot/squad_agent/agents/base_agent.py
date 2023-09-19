"""
All agents should inherit from this base class
"""

import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from os import path
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from loguru import logger
from sc2.data import Result
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from torch.utils.tensorboard import SummaryWriter

from bot.consts import SQUAD_ACTIONS, ConfigSettings

if TYPE_CHECKING:
    from ares import AresBot


class BaseAgent(metaclass=ABCMeta):
    __slots__ = (
        "ai",
        "config",
        "DATA_DIR",
        "device",
        "epoch",
        "current_action",
        "cumulative_reward",
        "squad_reward",
        "all_episode_data",
        "previous_close_enemy",
        "previous_main_squad",
        "writer",
        "action_distribution",
        "ml_training_file_path",
        "CHECKPOINT_PATH",
        "num_actions",
        "training_active",
        "axes",
        "fig",
        "visualize_spatial_features",
        "PLOT_TITLES",
    )

    def __init__(self, ai: "AresBot", config: Dict, device: str = "cpu"):
        super().__init__()
        self.ai: AresBot = ai
        self.config: Dict = config
        self.DATA_DIR: str = config[ConfigSettings.DATA_DIRECTORY]

        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using {self.device}")

        self.epoch: int = 0
        self.current_action = 0

        self.cumulative_reward: float = 0.0
        self.squad_reward: float = 0.0
        self.ml_training_file_path: str = path.join(
            self.DATA_DIR, "agent_training_history.json"
        )
        self.all_episode_data: List[Dict] = self.get_episode_data()
        self.previous_close_enemy: Optional[Units] = None
        self.previous_main_squad: Optional[Units] = None

        self.writer: SummaryWriter = SummaryWriter("data/runs")
        self.action_distribution: List[int] = [0 for _ in range(len(SQUAD_ACTIONS))]

        self.num_actions: int = len(SQUAD_ACTIONS)

        self.CHECKPOINT_PATH: str = path.join(
            self.DATA_DIR,
            config[ConfigSettings.SQUAD_AGENT][ConfigSettings.CHECKPOINT_NAME],
        )

        self.training_active: bool = not config[ConfigSettings.SQUAD_AGENT][
            ConfigSettings.INFERENCE_MODE
        ]
        logger.info(f"Training active: {self.training_active}")

        self.axes = None
        self.fig = None
        self.visualize_spatial_features: bool = config[
            ConfigSettings.VISUALIZE_SPATIAL_FEATURES
        ]

        self.PLOT_TITLES: list[str] = []

        if self.visualize_spatial_features:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("TkAgg")
            # self.fig = plt.figure(figsize=(10, 7))
            self.fig, self.axes = plt.subplots(2, 3)
            self.PLOT_TITLES = [
                "Enemy Influence",
                "Effects",
                "Height",
                "Creep",
                "Visibility",
                "Scatter (mean)",
            ]

    @property
    def reward(self) -> float:
        reward = self.squad_reward
        # clip the reward between -1 and 1 to help training
        reward = min(reward, 1.0) if reward >= 0.0 else max(reward, -1.0)
        return reward

    @abstractmethod
    def choose_action(
        self,
        squads: List,
        pos_of_squad: Point2,
        all_close_enemy: Units,
        squad_units: Units,
        attack_target: Point2,
        rally_point: Point2,
    ) -> int:
        self.previous_main_squad = squad_units
        self.previous_close_enemy = all_close_enemy
        return 0

    @abstractmethod
    def on_episode_end(self, result: Result) -> None:
        pass

    def on_unit_destroyed(self, tag: int) -> None:
        if self.previous_main_squad and tag in self.previous_main_squad.tags:
            unit: Unit = self.previous_main_squad.find_by_tag(tag)
            value = self.ai.calculate_unit_value(unit.type_id)
            value = (value.minerals + value.vespene * 1.5) / 700.0
            self.squad_reward -= value
        elif self.previous_close_enemy and tag in self.previous_close_enemy.tags:
            _unit: Unit = self.previous_close_enemy.find_by_tag(tag)
            value = self.ai.calculate_unit_value(_unit.type_id)
            value = (value.minerals + value.vespene * 1.5) / 700.0
            self.squad_reward += value

    def get_episode_data(self, get_default: bool = True) -> List[Dict]:
        if path.isfile(self.ml_training_file_path):
            with open(self.ml_training_file_path, "r") as f:
                episode_data = json.load(f)
        elif get_default:
            # no data, create a dummy version
            episode_data = [
                {
                    "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "GlobalStep": 0,
                    "Race": str(self.ai.enemy_race),
                    "Reward": 0.0,
                    "Result": 0,
                    "OppID": self.ai.opponent_id,
                    "ActionDistribution": [],
                    "MapName": self.ai.game_info.map_name,
                    "Loss": 0.0,
                }
            ]
        else:
            episode_data = []
        self.all_episode_data = episode_data
        return episode_data

    def store_episode_data(
        self, result, steps, reward, action_distribution, loss=0.0
    ) -> None:
        logger.info("Storing episode data")
        episode_data = self.get_episode_data(get_default=False)
        step = 0 if len(episode_data) == 0 else episode_data[-1]["GlobalStep"]

        result_id = self._get_result_id(result)
        episode_info = {
            "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "GlobalStep": steps + step,
            "Race": str(self.ai.enemy_race),
            "Reward": reward,
            "Result": result_id,
            "OppID": self.ai.opponent_id,
            "ActionDistribution": action_distribution,
            "MapName": self.ai.game_info.map_name,
            "Loss": loss,
        }
        if len(self.all_episode_data) >= 1:
            self.all_episode_data.append(episode_info)
        else:
            self.all_episode_data = [episode_info]
        with open(self.ml_training_file_path, "w") as f:
            json.dump(self.all_episode_data, f)

    @staticmethod
    def _get_result_id(result: Result) -> int:
        """
        Convert Result enum into an integer
        """
        return 2 if result == Result.Victory else (0 if result == Result.Defeat else 1)

    def _plot_spatial_features(self, spatial: torch.Tensor) -> None:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        images: list[torch.Tensor] = [
            spatial[0],  # influence
            spatial[1],  # effects
            spatial[2],  # height
            spatial[3],  # creep
            spatial[5],  # visibility
            spatial[6:].mean(0),  # scatter connections
        ]
        index: int = 0
        # A loop to access all subplots
        for i in range(2):
            for j in range(3):
                # assign the current subplot to the loop variable
                ax = self.axes[i, j]
                # clear the last figure
                ax.clear()
                # use the loop variable to assign title
                ax.set_title(self.PLOT_TITLES[index])

                # use the loop variable to assign the image
                ax.imshow(images[index])

                # remove axis
                ax.axis("off")

                index += 1

        plt.pause(0.0000000001)
        clear_output(wait=True)
