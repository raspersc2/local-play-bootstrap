from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import torch
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from torch import Tensor
from torch.nn.functional import one_hot

from bot.consts import BUFF_TYPES, UNIT_TYPES, ConfigSettings

if TYPE_CHECKING:
    from ares import AresBot

# since we one hot encode unit type ids
# we don't want to hot encode 1961+ different values
# so convert all unit type ids into our own index from consts
BUFF_TYPE_DICT = dict(zip(BUFF_TYPES, range(0, len(BUFF_TYPES))))
UNIT_TYPE_DICT = dict(zip(UNIT_TYPES, range(0, len(UNIT_TYPES))))

SPATIAL_SIZE = [152, 152]  # y, x
BUFF_LENGTH = 3
UPGRADE_LENGTH = 20
MAX_DELAY = 127
BEGINNING_ORDER_LENGTH = 20
MAX_SELECTED_UNITS_NUM = 64
EFFECT_LEN = 100

NUM_UNIT_TYPES: int = len(UnitTypeId)
NUM_BUFF_TYPES: int = len(BUFF_TYPES)
NUM_UPGRADES: int = len(UpgradeId)


class Features:
    def __init__(self, ai: "AresBot", config: dict, max_entities: int, device) -> None:
        self.ai: AresBot = ai
        self.units: List = []
        self.tags: List[int] = []
        self.max_entities: int = max_entities
        self.device = device
        self.map_size_y = self.ai.game_info.map_size.y
        self.visualize_spatial_features: bool = config[
            ConfigSettings.VISUALIZE_SPATIAL_FEATURES
        ]
        # height map remains static, only process it once
        height = self.ai.game_info.terrain_height.data_numpy.copy().T
        height = height[None, :]
        self.height: torch.Tensor = torch.from_numpy(height)

        self.fig = None
        if self.visualize_spatial_features:
            import matplotlib
            import matplotlib.pyplot as plt

            matplotlib.use("TkAgg")
            self.fig = plt.figure(figsize=(10, 7))

    def reset(self) -> None:
        self.units = []
        self.tags = []

    def append_unit(self, u, alliance: int, unit_type: int) -> None:
        if len(self.units) >= self.max_entities:
            return

        self.units.append(
            [
                UNIT_TYPE_DICT[unit_type],
                0 if alliance == 1 else 1,
                u.health,
                u.health_max,
                u.shield,
                u.shield_max,
                u.energy,
                u.energy_max,
                u.pos.x,
                self.map_size_y - u.pos.y,
                u.cloak,
                u.is_powered,
                u.weapon_cooldown,
                u.is_hallucination,
                BUFF_TYPE_DICT[u.buff_ids[0]] if len(u.buff_ids) >= 1 else 0,
                BUFF_TYPE_DICT[u.buff_ids[1]] if len(u.buff_ids) >= 2 else 0,
                u.is_active,
                u.attack_upgrade_level,
                u.armor_upgrade_level,
                u.shield_upgrade_level,
            ]
        )

    def transform_obs(
        self,
        ground_grid: np.ndarray,
        effects_grid: np.ndarray,
        pos_of_squad: Point2,
        attack_target: Point2,
        rally_point: Point2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        for unit in self.ai.state.observation_raw.units:
            alliance: int = unit.alliance
            unit_type: int = unit.unit_type
            self.append_unit(unit, alliance, unit_type)

        entity, entities_type, locations = self._process_entity_info()
        spatial = self._process_spatial_info(ground_grid, effects_grid)
        scalar: torch.Tensor = self._process_scalar_info(
            pos_of_squad, attack_target, rally_point
        )

        return spatial, entity, scalar, locations

    @staticmethod
    def _np_one_hot(targets: np.ndarray, nb_classes: int) -> np.ndarray:
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    def _process_entity_info(self):

        units: List = self.units[: self.max_entities]

        entities_array = np.array(units)

        entities_type = entities_array[:, 0].astype(np.int32)

        encoding_list = []

        unit_type_encoding = self._np_one_hot(entities_type, len(UNIT_TYPES) + 1)
        encoding_list.append(unit_type_encoding)

        alliance_encoding = self._np_one_hot(entities_array[:, 1].astype(np.int32), 2)
        encoding_list.append(alliance_encoding)

        health_ratio = np.expand_dims(
            entities_array[:, 2] / (entities_array[:, 3] + 1e-6), axis=1
        )
        shield_ratio = np.expand_dims(
            entities_array[:, 4] / (entities_array[:, 5] + 1e-6), axis=1
        )
        energy_ratio = np.expand_dims(
            entities_array[:, 6] / (entities_array[:, 7] + 1e-6), axis=1
        )
        x = np.expand_dims(entities_array[:, 8].astype(np.int32), axis=1)
        y = np.expand_dims(entities_array[:, 9].astype(np.int32), axis=1)
        encoding_list.extend(
            [
                health_ratio,
                shield_ratio,
                energy_ratio,
                x,  # x
                y,  # y
            ]
        )

        cloak_encoding = self._np_one_hot(entities_array[:, 10].astype(np.int32), 5)
        encoding_list.append(cloak_encoding)

        powered_encoding = self._np_one_hot(entities_array[:, 11].astype(np.int32), 2)
        encoding_list.append(powered_encoding)

        encoding_list.append(
            np.expand_dims(entities_array[:, 12], axis=1)
        )  # weapon cooldown

        halluc_encoding = self._np_one_hot(entities_array[:, 13].astype(np.int32), 2)
        encoding_list.append(halluc_encoding)

        buff_encoding1 = self._np_one_hot(
            entities_array[:, 14].astype(np.int32),
            NUM_BUFF_TYPES + 1,
        )
        buff_encoding2 = self._np_one_hot(
            entities_array[:, 15].astype(np.int32),
            NUM_BUFF_TYPES + 1,
        )
        encoding_list.extend([buff_encoding1, buff_encoding2])

        active_encoding = self._np_one_hot(entities_array[:, 16].astype(np.int32), 2)
        encoding_list.append(active_encoding)

        attack_upgrade_encoding = self._np_one_hot(
            entities_array[:, 17].astype(np.int32), 4
        )
        encoding_list.append(attack_upgrade_encoding)

        armor_upgrade_encoding = self._np_one_hot(
            entities_array[:, 18].astype(np.int32), 4
        )
        encoding_list.append(armor_upgrade_encoding)

        shield_upgrade_encoding = self._np_one_hot(
            entities_array[:, 19].astype(np.int32), 4
        )
        encoding_list.append(shield_upgrade_encoding)

        all_entities_array = np.concatenate(encoding_list, axis=1)

        all_entities_array = np.pad(
            all_entities_array,
            ((0, 256 - all_entities_array.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        all_entities_array = torch.from_numpy(all_entities_array)
        all_entities_array = all_entities_array.to(torch.float32)
        all_entities_array = torch.unsqueeze(all_entities_array, 0)

        locations = np.concatenate([x, y], axis=1)
        locations = np.pad(
            locations,
            ((0, 256 - locations.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        locations = torch.from_numpy(locations).to(torch.float32)
        locations = torch.unsqueeze(locations, 0)

        return all_entities_array, entities_type, locations

    def _process_spatial_info(
        self, ground_grid: np.ndarray, effects_grid: np.ndarray
    ) -> Tensor:

        spatial_arr: Union[list, Tensor] = []
        # location_grid = squad_grid.copy()

        ground_grid = ground_grid[None, :]
        ground_grid = torch.from_numpy(ground_grid)
        spatial_arr.append(ground_grid)

        effects_grid = effects_grid[None, :]
        effects_grid = torch.from_numpy(effects_grid)
        spatial_arr.append(effects_grid)

        spatial_arr.append(self.height)

        creep = self.ai.state.creep.data_numpy.T
        creep = torch.from_numpy(creep)
        creep = one_hot(creep.to(torch.int64), 2)
        creep = torch.movedim(creep, 2, 0)
        spatial_arr.append(creep)

        visibility = self.ai.state.visibility.data_numpy.T
        visibility = visibility[None, :]
        visibility = torch.tensor(visibility)
        spatial_arr.append(visibility)

        spatial_arr = torch.cat(spatial_arr)

        return spatial_arr

    def _process_scalar_info(
        self,
        pos_of_squad: Point2,
        attack_target: Point2,
        rally_point: Point2,
    ) -> torch.Tensor:
        """
        Basically any extra information that our agent may use
        """
        scalars: Union[list, Tensor, np.ndarray] = [
            pos_of_squad.x,
            self.map_size_y - pos_of_squad.y,
            self.ai.structures.amount,
            self.ai.enemy_structures.amount,
            attack_target.x,
            self.map_size_y - attack_target.y,
            rally_point.x,
            self.map_size_y - rally_point.y,
        ]
        scalars = np.array(scalars)
        scalars = torch.from_numpy(scalars)
        scalars = scalars.to(torch.float32)
        scalars = torch.unsqueeze(scalars, 0)
        return scalars
