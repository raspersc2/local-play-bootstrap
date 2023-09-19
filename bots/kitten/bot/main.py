from typing import Dict, List, Optional, Set, Tuple, Union

import yaml
from ares import AresBot
from ares.consts import ALL_STRUCTURES, UnitRole
from ares.dicts.unit_data import UNIT_DATA
from s2clientprotocol import raw_pb2 as raw_pb
from s2clientprotocol import sc2api_pb2 as sc_pb
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2, Point3
from sc2.unit import Unit
from sc2.units import Units
from scipy.spatial import KDTree

from bot.consts import AgentClass, ConfigSettings
from bot.modules.macro import Macro
from bot.modules.map_scouter import MapScouter
from bot.modules.workers import WorkersManager
from bot.squad_agent.agents.base_agent import BaseAgent
from bot.squad_agent.agents.dqn_agent import DQNAgent
from bot.squad_agent.agents.dqn_rainbow_agent import DQNRainbowAgent
from bot.squad_agent.agents.offline_agent import OfflineAgent
from bot.squad_agent.agents.ppo_agent import PPOAgent
from bot.squad_agent.agents.random_agent import RandomAgent
from bot.state import State
from bot.unit_squads import UnitSquads


class Kitten(AresBot):
    __slots__ = (
        "unit_squads",
        "workers_manager",
        "macro",
        "CONFIG_FILE",
        "agent_config",
        "debug",
        "sent_chat",
        "enemy_tree",
    )

    agent: BaseAgent
    macro: Macro
    unit_squads: UnitSquads

    def __init__(self) -> None:
        super().__init__()

        self.agent_config: Dict = dict()
        self.CONFIG_FILE = "config.yaml"
        with open(f"{self.CONFIG_FILE}", "r") as config_file:
            self.agent_config = yaml.safe_load(config_file)
        self.debug: bool = self.agent_config[ConfigSettings.DEBUG]

        self.map_scouter: MapScouter = MapScouter(self)

        self.workers_manager: WorkersManager = WorkersManager(self)
        self.sent_chat: bool = False
        self.enemy_tree: Optional[KDTree] = None

    async def on_start(self) -> None:
        await super(Kitten, self).on_start()

        # TODO: Improve this, handle invalid options in config and don't use if/else
        agent_class: str = self.agent_config[ConfigSettings.SQUAD_AGENT][
            ConfigSettings.AGENT_CLASS
        ]
        try:
            if agent_class == AgentClass.OFFLINE_AGENT:
                self.agent = OfflineAgent(self, self.agent_config)
            elif agent_class == AgentClass.PPO_AGENT:
                self.agent = PPOAgent(self, self.agent_config)
            elif agent_class == AgentClass.DQN_AGENT:
                self.agent = DQNAgent(self, self.agent_config)
            elif agent_class == AgentClass.DQN_RAINBOW_AGENT:
                self.agent = DQNRainbowAgent(self, self.agent_config)
            elif agent_class == AgentClass.RANDOM_AGENT:
                self.agent = RandomAgent(self, self.agent_config)
        except ValueError:
            raise ValueError("Invalid AgentClass name in config.yaml")

        self.macro = Macro(self, self.workers_manager, self.debug)
        self.unit_squads = UnitSquads(self, self.agent)
        self.client.game_step = self.agent_config[ConfigSettings.GAME_STEP]
        self.client.raw_affects_selection = True
        self.agent.get_episode_data()

        await self.map_scouter.initialize()

    async def on_step(self, iteration: int) -> None:
        await super(Kitten, self).on_step(iteration)
        if self.time > 1200.0:
            await self.client.leave()
        state: State = State(self)
        await self.unit_squads.update(iteration)
        await self.macro.update(state, iteration)
        self.workers_manager.update(state, iteration)
        self.map_scouter.update()

        if iteration % 16 == 0:
            for depot in self.structures(UnitTypeId.SUPPLYDEPOT):
                depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

        if self.time > 5.0 and not self.sent_chat:
            num_episodes: int = len(self.agent.all_episode_data)
            await self.chat_send(
                f"Meow! This kitty has trained for {num_episodes} episodes (happy)"
            )
            self.sent_chat = True

        if self.time > 179.0 and self.state.game_loop % 672 == 0:
            reward: float = self.agent.cumulative_reward
            emotion: str = (
                "meow" if reward == 0.0 else ("growl" if reward < 0.0 else "purr")
            )
            await self.chat_send(
                f"Cumulative episode reward: {round(reward, 4)} ...{emotion}"
            )

        if self.debug:
            height: float = self.get_terrain_z_height(self.mediator.get_own_nat)
            self.client.debug_text_world(
                "Own nat", Point3((*self.mediator.get_own_nat, height)), size=11
            )
            self.client.debug_text_world(
                "Enemy nat", Point3((*self.mediator.get_enemy_nat, height)), size=11
            )
            for unit in self.all_units:
                self.client.debug_text_world(f"{unit.tag}", unit, size=9)

    async def on_unit_created(self, unit: Unit) -> None:
        await super(Kitten, self).on_unit_created(unit)
        if unit.type_id not in {UnitTypeId.MULE, UnitTypeId.SCV}:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        await super(Kitten, self).on_unit_destroyed(unit_tag)
        self.agent.on_unit_destroyed(unit_tag)
        self.unit_squads.remove_tag(unit_tag)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super(Kitten, self).on_unit_took_damage(unit, amount_damage_taken)
        if not unit.is_structure:
            return

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            unit(AbilityId.CANCEL_BUILDINPROGRESS)

    async def on_end(self, game_result: Result) -> None:
        await super(Kitten, self).on_end(game_result)
        self.agent.on_episode_end(game_result)

    def enemies_in_range(self, units: Units, distance: float) -> Dict[int, Units]:
        """
        Get all enemies in range of multiple units in one call
        :param units:
        :param distance:
        :return: Dictionary: Key -> Unit tag, Value -> Units in range of that unit
        """
        if not self.enemy_tree or not self.enemy_units:
            return {units[index].tag: Units([], self) for index in range(len(units))}

        unit_positions: List[Point2] = [u.position for u in units]
        in_range_list: List[Units] = []
        if unit_positions:
            query_result = self.enemy_tree.query_ball_point(unit_positions, distance)
            for result in query_result:
                in_range_units = Units(
                    [self.enemy_units[index] for index in result], self
                )
                in_range_list.append(in_range_units)
        return {units[index].tag: in_range_list[index] for index in range(len(units))}

    @staticmethod
    def center_mass(units: Units, distance: float = 5.0) -> Tuple[Point2, int]:
        """
        :param units:
        :param distance:
        :return: Position where most units reside, num units at that position
        """
        center_mass: Point2 = units[0].position
        max_num_units: int = 0
        for unit in units:
            pos: Point2 = unit.position
            close: Units = units.closer_than(distance, pos)
            if len(close) > max_num_units:
                center_mass = pos
                max_num_units = len(close)

        return center_mass, max_num_units

    async def give_units_same_order(
        self,
        order: AbilityId,
        unit_tags: Union[List[int], Set[int]],
        target: Optional[Union[Point2, int]] = None,
    ) -> None:
        """
        Give units corresponding to the given tags the same order.
        @param order: the order to give to all units
        @param unit_tags: the tags of the units to give the order to
        @param target: either a Point2 of the location or the tag of the unit to target
        """
        if not target:
            # noinspection PyProtectedMember
            await self.client._execute(
                action=sc_pb.RequestAction(
                    actions=[
                        sc_pb.Action(
                            action_raw=raw_pb.ActionRaw(
                                unit_command=raw_pb.ActionRawUnitCommand(
                                    ability_id=order.value,
                                    unit_tags=unit_tags,
                                )
                            )
                        ),
                    ]
                )
            )
        elif isinstance(target, Point2):
            # noinspection PyProtectedMember
            await self.client._execute(
                action=sc_pb.RequestAction(
                    actions=[
                        sc_pb.Action(
                            action_raw=raw_pb.ActionRaw(
                                unit_command=raw_pb.ActionRawUnitCommand(
                                    ability_id=order.value,
                                    target_world_space_pos=target.as_Point2D,
                                    unit_tags=unit_tags,
                                )
                            )
                        ),
                    ]
                )
            )
        else:
            # noinspection PyProtectedMember
            await self.client._execute(
                action=sc_pb.RequestAction(
                    actions=[
                        sc_pb.Action(
                            action_raw=raw_pb.ActionRaw(
                                unit_command=raw_pb.ActionRawUnitCommand(
                                    ability_id=order.value,
                                    target_unit_tag=target,
                                    unit_tags=unit_tags,
                                )
                            )
                        ),
                    ]
                )
            )

    @staticmethod
    def get_total_supply(units: Units) -> int:
        """
        Get total supply of units.
        @param units:
        @return:
        """
        return sum(
            [
                UNIT_DATA[unit.type_id]["supply"]
                for unit in units
                # yes we did have a crash getting supply of a nuke!
                if unit.type_id not in ALL_STRUCTURES
                and unit.type_id != UnitTypeId.NUKE
            ]
        )
