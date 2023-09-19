"""
This script will play kitten vs other opponents to collect actions, rewards and states
When games have completed, the ppo_training script is invoked
This process is repeated

Steps:
1. Set up AI-Arena docker image https://github.com/aiarena/local-play-bootstrap
2. Move this python script to the root of `local-play-bootstrap` from the above repo
3. Copy the entire `kitten` folder into the `bots` folder in `local-play-bootstrap`
4. Find some other working bots from AI-Arena to train against: https://aiarena.net/bots/downloadable/
5. Configure this script, ensure maps path is correct and set up the matches
6. ???
7. Profit

Optional: Run Tensorboard pointing to `kitten/data/runs`
"""
# import time
from os import system
from pathlib import Path
from random import choice
# from bots.kitten.bot.squad_agent.training_scripts.ppo_trainer import PPOTrainer
# from bots.kitten.bot.squad_agent.training_scripts.update_dqn_target import update_dqn_target

MAP_FILE_EXT: str = "SC2Map"
MAPS_PATH: str = "./maps"

BOTS_PLAYER_ONE = [
    "kitten,T,python,",
]
BOTS_PLAYER_TWO = [
    "BluntCheese,Z,python,",
    "BluntFlies,Z,python,",
    "BluntMacro,Z,python,",
    "Kagamine,Z,java,",
    "RustyMarines,T,python,",
    "RustyOneBaseTurtle,T,python,",
    "VeTerran,T,cpplinux,",
    "MicroMachine,T,cpplinux,",
    "SharpRobots,P,python,",
    "SharpKnives,P,python,",
    "SharpCannons,P,python,",
    "TyrP,P,dotnetcore,",
]

NUM_GAMES_TO_PLAY: int = 1000000000000000000000000000000000

def get_random_string(length):
    # With combination of lower and upper case
    return ''.join(choice("asdflkghywuirre13y58!") for i in range(length))

if __name__ == "__main__":
    maps: list[str] = [
        p.name.replace(f".{MAP_FILE_EXT}", "")
        for p in Path(MAPS_PATH).glob(f"*.{MAP_FILE_EXT}")
        if p.is_file()
    ]
    print("matches started")
    for x in range(NUM_GAMES_TO_PLAY):
        matchString: str = (
            f"1,{choice(BOTS_PLAYER_ONE)}2,{choice(BOTS_PLAYER_TWO)}{choice(maps)}"
        )
        with open("matches", "w") as f:
            f.write(f"{matchString}")
        print(f"match {x}: {matchString}")
        system(f'cmd /c "docker compose -f docker-compose-host-network.yml up"')
        # ppo_trainer = PPOTrainer()
        # ppo_trainer.learn()
        # update_dqn_target()
    print("matches ended")
