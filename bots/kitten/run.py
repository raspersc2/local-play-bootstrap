import random
import sys

from sc2.bot_ai import BotAI

sys.path.append("ares-sc2/src/ares")
sys.path.append("ares-sc2/src")
sys.path.append("ares-sc2")

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import AIBuild, Bot, Computer

from bot.main import Kitten
from ladder import run_ladder_game


class DoNothingBot(BotAI):
    def __init__(self):
        super().__init__()

    async def on_start(self):
        pass

    async def on_step(self, iteration):
        pass


bot1 = Bot(Race.Terran, Kitten(), "kitten")
bot2 = Bot(Race.Zerg, DoNothingBot())


def main():
    # Ladder game started by LadderManager
    print("Starting ladder game...")
    result, opponentid = run_ladder_game(bot1)
    print(result, " against opponent ", opponentid)


# Start game
if __name__ == "__main__":
    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        random_map = random.choice(
            [
                # "BerlingradAIE",
                # "InsideAndOutAIE",
                # "MoondanceAIE",
                # "StargazersAIE",
                "WaterfallAIE",
                # "HardwireAIE",
            ]
        )
        random_race = random.choice([Race.Zerg, Race.Terran, Race.Protoss])
        print("Starting local game...")
        run_game(
            maps.get(random_map),
            [
                # bot2,
                bot1,
                Computer(random_race, Difficulty.CheatVision, ai_build=AIBuild.Macro),
                # bot2,
            ],
            realtime=False,
            save_replay_as="kitten.SC2Replay",
            # 2 lower spawn / 2564 upper spawn
            # random_seed=2,
        )
