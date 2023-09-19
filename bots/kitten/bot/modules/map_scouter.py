import itertools
from typing import TYPE_CHECKING, Any, Iterator

from ares.consts import UnitRole, UnitTreeQueryType
from ares.cython_extensions.units_utils import cy_closest_to
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

if TYPE_CHECKING:
    from ares import AresBot


class MapScouter:
    """
    Assign a marine to scout around the map so agent can pick up more data
    """

    expansions_generator: Iterator[Any]

    def __init__(self, ai: "AresBot") -> None:
        self.ai: AresBot = ai

        self.next_base_location: Point2 = Point2((1, 1))

        self.STEAL_FROM: set[UnitRole] = {UnitRole.ATTACKING}

    async def initialize(self) -> None:
        # set up the expansion generator,
        # so we can keep cycling through expansion locations
        base_locations: list[Point2] = [
            el[0] for el in self.ai.mediator.get_own_expansions[1:]
        ]
        self.expansions_generator = itertools.cycle(base_locations)
        self.next_base_location = next(self.expansions_generator)

    def update(self) -> None:
        if self.ai.time < 185.0:
            return

        existing_map_scouters: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.CONTROL_GROUP_ONE
        )

        if not existing_map_scouters:
            self._assign_map_scouter()
        else:
            for scout in existing_map_scouters:
                self._scout_map(scout)

    def _assign_map_scouter(self) -> None:
        if steal_from := self.ai.mediator.get_units_from_role(
            role=UnitRole.ATTACKING, unit_type=UnitTypeId.MARINE
        ):
            if len(steal_from) > 15:
                marine: Unit = cy_closest_to(self.ai.start_location, steal_from)
                self.ai.mediator.assign_role(
                    tag=marine.tag, role=UnitRole.CONTROL_GROUP_ONE
                )

    def _scout_map(self, scout: Unit) -> None:

        if self.next_base_location and self.ai.is_visible(self.next_base_location):
            self.next_base_location = next(self.expansions_generator)
            # skip location if we know about enemy here (skip once per step)
            enemy: list[Units] = self.ai.mediator.get_units_in_range(
                start_points=[scout.position],
                distances=[15.0],
                query_tree=UnitTreeQueryType.AllEnemy,
            )
            if len(enemy) > 0:
                self.next_base_location = next(self.expansions_generator)

        if scout.order_target != self.next_base_location:
            scout.move(self.next_base_location)
