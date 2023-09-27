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
from bots.kitten.bot.squad_agent.training_scripts.ppo_trainer import PPOTrainer
# from bots.kitten.bot.squad_agent.training_scripts.update_dqn_target import update_dqn_target
import boto3
import os
import platform

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
TRAIN_PPO: bool = False

LOCAL_DIRECTORY = './bots/kitten/data'
BUCKET_NAME = "kitten-spaces"

S3_CONFIG = {
    "region_name": "ams3",
    "endpoint_url": "https://ams3.digitaloceanspaces.com",
    "aws_access_key_id": "DO00RTCEYJ3DLEKFL2KK",
    "aws_secret_access_key": "LdSz28S4gkS8Ki3DA23aPF+3zBeWYfpQfpMnY9LWKh8"
}

def get_random_string(length):
    # With combination of lower and upper case
    return ''.join(choice("asdflkghywuirre13y58!") for _ in range(length))

def download_file_from_space(space_name, file_key, local_path):
    try:
        # Create an S3 client
        s3_client = boto3.client('s3', **S3_CONFIG)

        # Download the specific file from the Space
        s3_client.download_file(space_name, file_key, local_path)

        print(f"File '{file_key}' downloaded to '{local_path}' successfully.")
    except Exception as e:
        print(f"Error downloading file '{file_key}': {str(e)}")


if __name__ == "__main__":
    # Create an S3 client
    s3_client = boto3.client('s3', **S3_CONFIG)

    file_key_to_download = "checkpoint.pt"

    local_path_to_save = f"{LOCAL_DIRECTORY}/checkpoint.pt"

    download_file_from_space(BUCKET_NAME, file_key_to_download, local_path_to_save)

    # try:
    #     # List objects (files and directories) within the specified Space
    #     response = s3_client.list_objects_v2(Bucket=space_name)
    #
    #     if 'Contents' in response:
    #         print(f"Contents of '{space_name}':")
    #         for obj in response['Contents']:
    #             print(obj['Key'])
    #     else:
    #         print(f"'{space_name}' is empty.")
    # except Exception as e:
    #     print(f"Error listing objects in '{space_name}': {str(e)}")



    # for root, dirs, files in os.walk(LOCAL_DIRECTORY):
    #     if "runs" in dirs:
    #         dirs.remove("runs")
    #
    #     for file in files:
    #         # no need to upload the model
    #         # if file.startswith("checkpoint") or file.endswith(".lock"):
    #         #     continue
    #
    #         local_file = os.path.join(root, file)
    #         # Calculate the object key relative to the directory
    #         relative_key = os.path.relpath(local_file, LOCAL_DIRECTORY)
    #         # Replace backslashes (if running on Windows)
    #         if platform.system() == "Windows":
    #             relative_key = relative_key.replace("\\", "/")
    #
    #         # Upload the file to the Space using the relative key
    #         s3_client.upload_file(local_file, BUCKET_NAME, relative_key)

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
        system("docker compose -f docker-compose-host-network.yml up")
        if TRAIN_PPO:
            ppo_trainer = PPOTrainer()
            ppo_trainer.learn()
        else:
            pass
        # update_dqn_target()
    print("matches ended")
