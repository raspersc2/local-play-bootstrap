from typing import TYPE_CHECKING, Dict, Set, Union

from ares.cython_extensions.geometry import cy_distance_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

if TYPE_CHECKING:
    from ares import AresBot


class UnitSquad:
    __slots__ = (
        "ai",
        "squad_id",
        "squad_position",
        "squad_units",
        "current_action",
        "current_action_position",
        "stutter_forward",
        "stuttering",
        "stim_next_step",
        "stim_locked_till",
        "STIM_DURATION",
        "action_locked_till",
    )

    def __init__(self, ai: "AresBot", squad_id: str, squad_units: Units):
        self.ai: AresBot = ai
        self.squad_id: str = squad_id
        self.squad_units: Units = squad_units
        self.squad_position: Point2 = squad_units.center

        self.current_action: AbilityId = AbilityId.ATTACK
        self.current_action_position: Point2 = self.ai.game_info.map_center
        # if set to False, units will kite back by default
        self.stutter_forward: bool = False
        # if we store this, then the stutter action is only issued once
        self.stuttering: bool = False

        self.stim_next_step: bool = False
        self.stim_locked_till: float = 0.0
        self.STIM_DURATION: float = 11.0
        # non-main squads get locked for a short period
        self.action_locked_till: float = 0.0

    def set_stim_status(self, status: bool) -> None:
        # Only one stim per self.STIM_DURATION seconds
        if status and (
            self.ai.time < self.stim_locked_till
            or UpgradeId.STIMPACK not in self.ai.state.upgrades
        ):
            self.stim_next_step = False
            return

        self.stim_next_step = status

    def set_squad_units(self, units: Units) -> None:
        self.squad_units = units

    def update_action(
        self, action: AbilityId, position: Point2, stutter_forward: bool = False
    ) -> None:
        self.current_action = action
        self.current_action_position = position
        self.stutter_forward = stutter_forward

    async def do_action(self, squad_tags: Set[int], main_squad: bool = False) -> None:
        # currently only main squad uses RL agent, all other squads have scripted logic
        if not main_squad:
            if self.ai.time > self.action_locked_till or self.squad_units(
                UnitTypeId.MEDIVAC
            ):
                await self._do_scripted_squad_action(squad_tags)
        else:
            if self.stim_next_step:
                self.stim_next_step = False
                self.stim_locked_till = self.ai.time + self.STIM_DURATION
                await self.ai.give_units_same_order(AbilityId.EFFECT_STIM, squad_tags)
            else:
                if self.current_action == AbilityId.HOLDPOSITION:
                    perform_action: bool = False
                    for unit in self.squad_units:
                        if not unit.is_using_ability(AbilityId.HOLDPOSITION):
                            perform_action = True
                            break
                    if perform_action:
                        await self.ai.give_units_same_order(
                            AbilityId.HOLDPOSITION, squad_tags
                        )
                elif self.current_action == AbilityId.ATTACK:
                    await self._do_squad_attack_action(squad_tags)
                else:
                    await self._do_squad_move_action(squad_tags)

    async def _do_scripted_squad_action(self, squad_tags: Set[int]) -> None:
        """
        Used for smaller squads
        Main goal is to join up with the main squad
        Without spamming too many actions
        """
        unit: Unit = self.squad_units[0]
        target: Union[Point2, int, None] = unit.order_target
        if (
            target
            and isinstance(target, Point2)
            and cy_distance_to(target.position, self.current_action_position) < 8.0
        ):
            return

        await self.ai.give_units_same_order(
            AbilityId.ATTACK, squad_tags, self.current_action_position
        )
        self.action_locked_till = self.ai.time + 5.0

    def should_stutter(self, close_enemy: Dict[int, Units]) -> bool:
        avg_weapon_cooldown: float = sum(
            [u.weapon_cooldown for u in self.squad_units]
        ) / len(self.squad_units)
        # all weapons are ready, should stay on attack command
        if avg_weapon_cooldown == 0.0:
            return False

        if self.stutter_forward:
            # if all units are in range of something, don't worry about moving
            all_in_range: bool = True
            for unit in self.squad_units:
                enemy: Units = close_enemy[unit.tag]
                if not enemy.in_attack_range_of(unit):
                    all_in_range = False
                    break
            if all_in_range:
                return False
            return avg_weapon_cooldown > 3.5
        else:
            return avg_weapon_cooldown > 5.5

    async def _do_squad_attack_action(self, squad_tags: Set[int]) -> None:
        close_enemy: Dict[int, Units] = self.ai.enemies_in_range(self.squad_units, 15.0)
        should_stutter: bool = self.should_stutter(close_enemy)
        sample_unit: Unit = self.squad_units[0]
        # all units weapons are ready, a-move to current action position
        # TODO: Add some target fire, keeping in mind APM?
        if not should_stutter and not sample_unit.is_attacking:
            self.stuttering = False
            await self.ai.give_units_same_order(
                AbilityId.ATTACK,
                squad_tags,
                self.current_action_position,
            )

        # else move command depending on the agent's action type
        elif should_stutter and not self.stuttering:
            # only call this once till weapons are ready again, so action isn't spammed
            self.stuttering = True
            pos: Point2
            if self.stutter_forward:
                center: Point2 = self.squad_units.center
                if close_enemies := self.ai.enemy_units.filter(
                    lambda u: u.distance_to(center)
                ):
                    pos = close_enemies.center
                else:
                    pos = self.current_action_position
            else:
                # get a path back home, and kite back using that
                pos = self.ai.mediator.find_path_next_point(
                    start=self.squad_position,
                    target=self.ai.start_location,
                    grid=self.ai.mediator.get_ground_grid,
                    sensitivity=8,
                )
            await self.ai.give_units_same_order(AbilityId.MOVE, squad_tags, pos)

    async def _do_squad_move_action(self, squad_tags: Set[int]) -> None:
        if cy_distance_to(self.squad_position, self.current_action_position) < 3.0:
            return

        sample_unit: Unit = self.squad_units[0]
        order_target: Union[int, Point2, None] = sample_unit.order_target

        pos: Point2 = self.ai.mediator.find_path_next_point(
            start=self.squad_position,
            target=self.current_action_position,
            grid=self.ai.mediator.get_ground_grid,
            sensitivity=6,
        )
        # sample unit is already close to the calculated position
        if (
            order_target
            and isinstance(order_target, Point2)
            and cy_distance_to(sample_unit.position, pos) < 2.5
        ):
            return

        # the pos we calculated is not that different to previous target
        if (
            order_target
            and isinstance(order_target, Point2)
            and cy_distance_to(order_target, pos) < 4.5
        ):
            return

        await self.ai.give_units_same_order(self.current_action, squad_tags, pos)
