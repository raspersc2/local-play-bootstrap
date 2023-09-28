"""
This script is meant to run on a cloud service provider
Specifically this was set up to run with digital ocean droplets / spaces bucket
The VPS should be setup to run aiarena-docker

Logic:
for each game:
    fetch_model_from_bucket()
    # use model to choose actions
    play_game()
    collect_states_actions_rewards_from_game()
    send_data_to_bucket()
    cleanup_data()
"""
import platform
import boto3
from os import listdir, path, remove, system, walk, makedirs
from pathlib import Path
from random import choice
import shutil
from s3_config import S3_CONFIG

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

NUM_GAMES_TO_PLAY: int = 10000000000000000

LOCAL_DIRECTORY = './bots/kitten/data'
BUCKET_NAME = "kitten-spaces"
MODEL_NAME = "checkpoint.pt"

def clean_up(random_dir_name):
    """
    Clear out kitten's data directory
    """
    files = listdir(LOCAL_DIRECTORY)

    # Loop through the files and remove them
    for file in files:
        if file == "history.json":
            continue
        file_path = path.join(LOCAL_DIRECTORY, file)
        if path.isfile(file_path):
            remove(file_path)

    if path.exists(f"{LOCAL_DIRECTORY}/states"):
        shutil.rmtree(f"{LOCAL_DIRECTORY}/states")
    if path.exists(f"{LOCAL_DIRECTORY}/uploading_{random_dir_name}"):
        shutil.rmtree(f"{LOCAL_DIRECTORY}/uploading_{random_dir_name}")

def copy_and_rename_directory(client, old_directory, new_directory):
    # List objects in the old directory
    objects = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=old_directory)['Contents']

    for obj in objects:
        # Calculate the new object key by replacing the old directory with the new directory
        new_key = new_directory + obj['Key'][len(old_directory):]

        # Copy the object to the new key
        client.copy_object(
            Bucket=BUCKET_NAME,
            CopySource={'Bucket': BUCKET_NAME, 'Key': obj['Key']},
            Key=new_key
        )

        # Delete the old object
        client.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])

    # Delete the old directory
    client.delete_object(Bucket=BUCKET_NAME, Key=old_directory)

def get_random_string(length: int = 32) -> str:
    # With combination of lower and upper case
    return ''.join(choice("asdflkghywuirze13y58TOJBPASZ") for _ in range(length))

def download_file_from_space(client, space_name, file_key, local_path) -> bool:
    try:
        # Download the specific file from the Space
        client.download_file(space_name, file_key, local_path)

        print(f"File '{file_key}' downloaded to '{local_path}' successfully.")
        return True
    except Exception as e:
        print(f"Error downloading file '{file_key}': {str(e)}")
        return False

def upload_files_to_space(client, random_dir_name):

    dir_name = f"uploading_{random_dir_name}"

    destination_directory = path.join(LOCAL_DIRECTORY, dir_name)
    makedirs(destination_directory)
    # Move all files from LOCAL_DIRECTORY to the destination_directory
    for item in listdir(LOCAL_DIRECTORY):
        source_item = path.join(LOCAL_DIRECTORY, item)
        destination_item = path.join(destination_directory, item)

        if item == dir_name or item.startswith("checkpoint") or item == "history.json":
            continue

        if path.isdir(source_item):
            # Use shutil.copytree to copy the entire subdirectory and its contents
            shutil.copytree(source_item, destination_item)
            # Remove the original subdirectory and its contents
            # shutil.rmtree(source_item)
        elif path.isfile(source_item):
            # Move individual files
            shutil.move(source_item, destination_item)


    for root, dirs, files in walk(destination_directory):
        if "runs" in dirs:
            dirs.remove("runs")

        for file in files:
            # no need to upload the model
            if file.startswith("checkpoint") or file.endswith(".lock"):
                continue

            local_file = path.join(root, file)
            # Calculate the object key relative to the directory
            relative_key = path.relpath(local_file, LOCAL_DIRECTORY)
            # Replace backslashes (if running on Windows)
            if platform.system() == "Windows":
                relative_key = relative_key.replace("\\", "/")

            # Add the random directory name as a prefix to the relative_key
            # new_relative_key = path.join(dir_name, relative_key)
            # Upload the file to the Space using the relative key
            try:
                client.upload_file(local_file, BUCKET_NAME, relative_key)
                print(f"File '{local_file}' uploaded to '{BUCKET_NAME}' successfully.")
            except Exception as e:
                print(f"Error uploading {local_file} to path {relative_key} at {BUCKET_NAME}: {str(e)}")

    # rename directory to remove the 'uploading_' prefix
    copy_and_rename_directory(client, dir_name, random_dir_name)

if __name__ == "__main__":
    # Create an S3 client
    s3_client = boto3.client('s3', **S3_CONFIG)
    local_path_to_save = f"{LOCAL_DIRECTORY}/{MODEL_NAME}"

    maps: list[str] = [
        p.name.replace(f".{MAP_FILE_EXT}", "")
        for p in Path(MAPS_PATH).glob(f"*.{MAP_FILE_EXT}")
        if p.is_file()
    ]
    print("matches started")
    for x in range(NUM_GAMES_TO_PLAY):
        # get latest model from our bucket, only continue if this succeeds
        # (don't want ai acting on incorrect model)
        if download_file_from_space(s3_client, BUCKET_NAME, MODEL_NAME, local_path_to_save):
            matchString: str = (
                f"1,{choice(BOTS_PLAYER_ONE)}2,{choice(BOTS_PLAYER_TWO)}{choice(maps)}"
            )
            with open("matches", "w") as f:
                f.write(f"{matchString}")
            print(f"match {x}: {matchString}")
            # play game
            system("docker compose -f docker-compose-host-network.yml up")
            # upload states to bucket for learning
            states_path = path.join(LOCAL_DIRECTORY, "states")
            if path.exists(states_path) and path.isdir(states_path):
                random_dir_name = get_random_string()
                upload_files_to_space(s3_client, random_dir_name)
                clean_up(random_dir_name)

    print("matches ended")
