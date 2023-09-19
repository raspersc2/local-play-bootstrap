from typing import TYPE_CHECKING, Set

from ares.behaviors.macro import Mining
from ares.consts import UnitRole
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit
from sc2.units import Units

from bot.consts import WORKERS_DEFEND_AGAINST
from bot.state import State

if TYPE_CHECKING:
    from ares import AresBot


class WorkersManager:
    __slots__ = (
        "ai",
        "workers_per_gas",
        "enemy_committed_worker_rush",
        "worker_defence_tags",
        "issued_scout_commands",
    )

    def __init__(self, ai: "AresBot") -> None:
        self.ai: AresBot = ai
        self.workers_per_gas: int = 3
        self.enemy_committed_worker_rush: bool = False
        self.worker_defence_tags: Set = set()
        self.issued_scout_commands: bool = False

    def update(self, state: State, iteration: int) -> None:
        self.ai.register_behavior(Mining(mineral_boost=False, keep_safe=False))

        for oc in state.orbitals.filter(lambda x: x.energy >= 50):
            mfs: Units = self.ai.mineral_field.closer_than(10, oc)
            if mfs:
                mf: Unit = max(mfs, key=lambda x: x.mineral_contents)
                oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)

        self._handle_worker_rush()

        self._handle_worker_scout()

    def _handle_worker_rush(self) -> None:
        """zerglings too !"""
        # got to a point in time we don't care about this anymore
        # scvs should go idle, at which point the
        # gathering resources logic should kick in
        if (
            self.ai.time > 200.0 and not self.enemy_committed_worker_rush
        ) or not self.ai.workers:
            self.worker_defence_tags = set()
            return

        enemy_workers: Units = self.ai.enemy_units.filter(
            lambda u: u.type_id in WORKERS_DEFEND_AGAINST
            and (u.distance_to(self.ai.start_location) < 25.0)
        )
        enemy_lings: Units = enemy_workers(UnitTypeId.ZERGLING)

        if enemy_workers.amount > 8 and self.ai.time < 180:
            self.enemy_committed_worker_rush = True

        # calculate how many workers we should use to defend
        num_enemy_workers: int = enemy_workers.amount
        if num_enemy_workers > 0:
            workers_needed: int = (
                num_enemy_workers
                if num_enemy_workers <= 6 and enemy_lings.amount <= 3
                else len(self.ai.workers)
            )
            if len(self.worker_defence_tags) < workers_needed:
                workers_to_take: int = workers_needed - len(self.worker_defence_tags)
                unassigned_workers: Units = self.ai.workers.tags_not_in(
                    self.worker_defence_tags
                )
                if workers_to_take > 0:
                    workers: Units = unassigned_workers.take(workers_to_take)
                    for worker in workers:
                        self.worker_defence_tags.add(worker.tag)
                        self.ai.mediator.assign_role(
                            tag=worker.tag, role=UnitRole.DEFENDING
                        )

        # actually defend if there is a worker threat
        if len(self.worker_defence_tags) > 0 and self.ai.mineral_field:
            defence_workers: Units = self.ai.workers.tags_in(self.worker_defence_tags)
            close_mineral_patch: Unit = self.ai.mineral_field.closest_to(
                self.ai.start_location
            )
            if defence_workers and enemy_workers:
                for worker in defence_workers:
                    if worker.weapon_cooldown == 0 and worker.is_attacking:
                        continue

                    # in attack range of enemy, prioritise attacking
                    if (
                        worker.weapon_cooldown == 0
                        and enemy_workers.in_attack_range_of(worker)
                    ):
                        worker.attack(enemy_workers.closest_to(worker))
                    # attack the workers
                    elif worker.weapon_cooldown == 0 and enemy_workers:
                        worker.attack(enemy_workers.closest_to(worker))
                    else:
                        worker.gather(close_mineral_patch)
            elif defence_workers:
                for worker in defence_workers:
                    worker.gather(close_mineral_patch)
                    self.ai.mediator.assign_role(
                        tag=worker.tag, role=UnitRole.GATHERING
                    )
                self.worker_defence_tags = set()

    def _handle_worker_scout(self) -> None:
        if self.ai.time > 20.0 and not self.issued_scout_commands:
            if worker := self.ai.mediator.select_worker(
                target_position=self.ai.start_location
            ):
                self.issued_scout_commands = True
                worker.move(
                    self.ai.enemy_start_locations[0].towards(
                        self.ai.game_info.map_center, -9.0
                    )
                )
                for el in self.ai.mediator.get_own_expansions:
                    worker.move(el[0], queue=True)

                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.SCOUTING)

        if scouts := self.ai.mediator.get_units_from_role(role=UnitRole.SCOUTING):
            for scout in scouts:
                if scout.is_idle:
                    self.ai.mediator.assign_role(tag=scout.tag, role=UnitRole.GATHERING)
