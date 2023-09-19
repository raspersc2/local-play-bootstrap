from dataclasses import dataclass
from typing import Optional

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.units import Units

"""
Used to store things we require more then once
To save iterations retrieving them again
"""


@dataclass
class State:
    ai: BotAI
    barracks: Optional[Units] = None
    main_build_area: Optional[Point2] = None
    natural_build_area: Optional[Point2] = None
    ccs: Optional[Units] = None
    depots: Optional[Units] = None
    factories: Optional[Units] = None
    orbitals: Optional[Units] = None
    starports: Optional[Units] = None

    def __post_init__(self):
        self.barracks = self.ai.structures(UnitTypeId.BARRACKS)

        self.ccs = self.ai.townhalls(UnitTypeId.COMMANDCENTER)
        self.depots = self.ai.structures(UnitTypeId.SUPPLYDEPOT)
        self.orbitals = self.ai.townhalls(UnitTypeId.ORBITALCOMMAND)
        self.factories = self.ai.structures(
            {UnitTypeId.FACTORY, UnitTypeId.FACTORYFLYING}
        )
        self.starports = self.ai.structures(UnitTypeId.STARPORT)

        if self.ai.townhalls:
            self.natural_build_area: Point2 = self.ai.townhalls.furthest_to(
                self.ai.start_location
            ).position.towards(self.ai.game_info.map_center, 5.5)

        if len(self.barracks) < 5:
            self.main_build_area: Point2 = self.ai.start_location.towards(
                self.ai.game_info.map_center, 7.0
            )
        else:
            self.main_build_area: Point2 = self.ai.start_location.towards(
                self.ai.game_info.map_center, 12.5
            )
