"""
Basic macro that focuses on bio production and upgrades
This should probably be rewritten /
    refactored into separate files for anything more complicated
"""
from typing import TYPE_CHECKING, Union

from ares.behaviors.macro import AutoSupply, BuildStructure, SpawnController
from ares.consts import UnitRole
from ares.cython_extensions.units_utils import cy_closest_to
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2, Pointlike
from sc2.unit import Unit
from sc2.units import Units

from bot.modules.workers import WorkersManager
from bot.state import State

if TYPE_CHECKING:
    from ares import AresBot


class Macro:
    __slots__ = (
        "ai",
        "workers_manager",
        "debug",
        "state",
        "max_workers",
    )

    state: State

    def __init__(
        self,
        ai: "AresBot",
        workers_manager: WorkersManager,
        debug: bool,
    ):
        self.ai: AresBot = ai
        self.workers_manager: WorkersManager = workers_manager
        self.debug: bool = debug
        self.max_workers: int = 19

    async def update(self, state: State, iteration: int) -> None:
        self.state = state

        tags_received_action: set[int] = await self._build_addons()
        self.ai.register_behavior(
            SpawnController(
                army_composition_dict={
                    UnitTypeId.MARINE: {
                        "proportion": 0.6,
                        "priority": 2,
                    },  # lowest priority
                    UnitTypeId.MEDIVAC: {"proportion": 0.25, "priority": 1},
                    UnitTypeId.MARAUDER: {
                        "proportion": 0.15,
                        "priority": 0,
                    },  # highest priority
                },
                freeflow_mode=True,
                ignored_build_from_tags=tags_received_action,
            )
        )

        if not self.ai.build_order_runner.build_completed:
            return

        if len(self.ai.townhalls) > 1:
            self.max_workers = 41

        self._build_factory()
        self._build_starport()

        self._manage_upgrades()
        self._build_refineries()
        self.ai.register_behavior(AutoSupply(self.state.main_build_area))
        self._produce_workers()
        self._build_barracks()
        self._build_bays()

        # 2 townhalls at all times
        if (
            len(self.ai.townhalls) < 2
            and self.ai.can_afford(UnitTypeId.COMMANDCENTER)
            and self._pending_structures(UnitTypeId.COMMANDCENTER) == 0
        ):
            if location := await self.ai.get_next_expansion():
                if worker := self.ai.mediator.select_worker(
                    target_position=self.ai.start_location
                ):
                    self.ai.mediator.build_with_specific_worker(
                        worker=worker,
                        structure_type=UnitTypeId.COMMANDCENTER,
                        pos=location,
                    )

    def _produce_workers(self) -> None:
        if (
            self.ai.supply_workers >= self.max_workers
            or not self.ai.can_afford(UnitTypeId.SCV)
            or self.ai.supply_left <= 0
            or not self.ai.townhalls.idle
        ):
            return

        # no rax yet, all ths can build scvs
        if self.state.barracks.ready.amount < 1:
            for th in self.ai.townhalls.idle:
                if th.is_ready:
                    th.train(UnitTypeId.SCV)
        # rax present, only orbitals / pfs can build scvs
        # TODO: Adjust this if we build PFs
        else:
            for th in self.state.orbitals.idle:
                th.train(UnitTypeId.SCV)

    def _pending_structures(self, structure_type: UnitTypeId) -> int:
        return (
            int(self.ai.already_pending(structure_type))
            + self.ai.mediator.get_building_counter[structure_type]
        )

    async def _build_addons(self) -> set[int]:
        tags_received_action: set[int] = set()
        ready_rax: Units = self.state.barracks.filter(lambda u: u.is_ready)
        if len(ready_rax) < 3 or self.ai.vespene < 25:
            return tags_received_action

        add_ons: list[Unit] = self.ai.mediator.get_own_structures_dict[
            UnitTypeId.BARRACKSTECHLAB
        ]
        max_add_ons: int = 2 if len(self.state.barracks) > 5 else 1
        if len(add_ons) < max_add_ons and self.ai.can_afford(UnitTypeId.TECHLAB):
            rax: Units = ready_rax.filter(lambda u: u.is_idle)
            for b in rax:
                if not b.has_add_on:
                    b.build(UnitTypeId.STARPORTTECHLAB)
                    tags_received_action.add(b.tag)

        return tags_received_action

    def _build_barracks(self) -> None:
        max_barracks: int = (
            2 if len(self.ai.townhalls) <= 1 else (4 if not self.state.factories else 8)
        )
        if self.ai.minerals > 500:
            max_barracks = 9
        max_pending: int = 1 if self.ai.minerals < 400 else 5
        rax: Units = self.state.barracks
        if self._dont_build(
            rax,
            UnitTypeId.BARRACKS,
            num_existing=max_barracks,
            max_pending=max_pending,
        ):
            return

        self.ai.register_behavior(
            BuildStructure(self.state.main_build_area, UnitTypeId.BARRACKS)
        )

    def _manage_upgrades(self) -> None:
        ccs: list[Unit] = [cc for cc in self.state.ccs if cc.build_progress == 1.0 and cc.is_idle]
        if (
            len(ccs) > 0
            and self.ai.can_afford(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)
        ):
            ccs[0](AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)

        if not self.state.factories:
            return

        idle_rax_tech_labs: list[Unit] = [
            tl
            for tl in self.ai.mediator.get_own_structures_dict[
                UnitTypeId.BARRACKSTECHLAB
            ]
            if tl.is_idle
        ]
        if len(idle_rax_tech_labs) > 0:
            if self.ai.already_pending_upgrade(
                UpgradeId.SHIELDWALL
            ) == 0 and self.ai.can_afford(UpgradeId.SHIELDWALL):
                self.ai.research(UpgradeId.SHIELDWALL)
                return
            # if self.ai.already_pending_upgrade(
            #     UpgradeId.STIMPACK
            # ) == 0 and self.ai.can_afford(UpgradeId.STIMPACK):
            #     self.ai.research(UpgradeId.STIMPACK)
            #     return
            if (
                self.ai.already_pending_upgrade(UpgradeId.PUNISHERGRENADES) == 0
                and self.ai.can_afford(UpgradeId.PUNISHERGRENADES)
                # and UpgradeId.STIMPACK in self.ai.state.upgrades
                and UpgradeId.SHIELDWALL in self.ai.state.upgrades
            ):
                self.ai.research(UpgradeId.PUNISHERGRENADES)
                return

        if self.ai.already_pending_upgrade(
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1
        ) == 0 and self.ai.can_afford(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1):
            self.ai.research(UpgradeId.TERRANINFANTRYWEAPONSLEVEL1)
            return

        if self.ai.already_pending_upgrade(
            UpgradeId.TERRANINFANTRYARMORSLEVEL1
        ) == 0 and self.ai.can_afford(UpgradeId.TERRANINFANTRYARMORSLEVEL1):
            self.ai.research(UpgradeId.TERRANINFANTRYARMORSLEVEL1)
            return

    def _build_refineries(self) -> None:
        if len(self.ai.gas_buildings) == 2:
            return

        pending = self._pending_structures(UnitTypeId.REFINERY)
        if pending:
            return

        num_rax: int = len(self.state.barracks)
        max_gas: int = 2 if num_rax >= 3 else (1 if num_rax >= 2 else 0)
        current_gas_num = pending + self.ai.gas_buildings.amount
        if current_gas_num < max_gas and self.ai.can_afford(UnitTypeId.REFINERY):
            if worker := self.ai.mediator.select_worker(
                target_position=self.ai.start_location
            ):
                geysers: Units = self.ai.vespene_geyser.filter(
                    lambda vg: not self.ai.gas_buildings.closer_than(2, vg)
                )
                self.ai.mediator.build_with_specific_worker(
                    worker=worker,
                    structure_type=UnitTypeId.REFINERY,
                    pos=cy_closest_to(self.ai.start_location, geysers),
                )
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)

    def _build_factory(self) -> None:
        factories: Units = self.state.factories

        # we only care about factories for the starport
        if self.state.starports:
            return

        if self._dont_build(factories, UnitTypeId.FACTORY):
            return

        self.ai.register_behavior(
            BuildStructure(self.state.main_build_area, UnitTypeId.FACTORY)
        )

    def _build_starport(self) -> None:
        ports: Units = self.state.starports
        if self._dont_build(ports, UnitTypeId.STARPORT):
            return

        self.ai.register_behavior(
            BuildStructure(self.state.main_build_area, UnitTypeId.STARPORT)
        )

    def _dont_build(
        self,
        structures: Union[Units, list[Unit]],
        structure_type: UnitTypeId,
        num_existing: int = 1,
        max_pending: int = 1,
    ) -> bool:
        return (
            self.ai.tech_requirement_progress(structure_type) != 1
            or len(structures) >= num_existing
            or self._pending_structures(structure_type) >= max_pending
            or self.ai.calculate_cost(structure_type).minerals > self.ai.minerals - 75
            or (
                self.ai.calculate_cost(structure_type).vespene > self.ai.vespene - 25
                and structure_type != UnitTypeId.BARRACKS
            )
        )

    def _build_bays(self) -> None:
        bay_type: UnitTypeId = UnitTypeId.ENGINEERINGBAY
        bays: list[Unit] = self.ai.mediator.get_own_structures_dict[bay_type]

        if self._dont_build(bays, bay_type) or len(self.state.starports) < 1:
            return

        self.ai.register_behavior(
            BuildStructure(self.state.main_build_area, UnitTypeId.ENGINEERINGBAY)
        )
