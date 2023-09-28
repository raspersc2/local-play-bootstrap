"""
Watch a S3 instance and train on new data
In this case we are using digital ocean spaces
We will download data from spaces and put it in kitten's data directory, since
PPOTrainer is designed to work with that in mind
After downloading -> train on data and update model
"""
import time

from bots.kitten.bot.squad_agent.training_scripts.ppo_trainer import PPOTrainer
import boto3
import json
import os
from s3_config import S3_CONFIG

LOCAL_DIRECTORY = './bots/kitten/data'
LOCAL_MASTER_JSON_FILE = os.path.join(LOCAL_DIRECTORY, "history.json")
BUCKET_NAME = "kitten-spaces"
MODEL_NAME = "checkpoint.pt"

def append_json_to_master(remote_json_file, local_master_json_file):
    # Load the contents of the downloaded JSON file
    if os.path.exists(local_master_json_file):
        with open(local_master_json_file, 'r') as master_file:
            master_data = json.load(master_file)
    else:
        return

    if os.path.exists(remote_json_file):
        # Load the contents of the JSON file from the bucket
        with open(remote_json_file, 'r') as remote_file:
            remote_data = json.load(remote_file)
    else:
        return

    # Append the data from the remote JSON file to the master JSON data
    master_data.extend(remote_data)

    # Write the updated master JSON data back to the local master file
    with open(local_master_json_file, 'w') as master_file:
        json.dump(master_data, master_file)

def delete_directory_and_contents(client, directory_to_delete):
    """
    Clear out data from the bucket that we are finished with
    """
    # List objects in the directory to delete
    objects = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=directory_to_delete)['Contents']

    # Delete each object in the directory
    for obj in objects:
        client.delete_object(Bucket=BUCKET_NAME, Key=obj['Key'])

    # Delete the directory itself
    client.delete_object(Bucket=BUCKET_NAME, Key=directory_to_delete)


if __name__ == "__main__":
    s3_client = boto3.client('s3', **S3_CONFIG)

    first_iteration: bool = True

    last_seen_objects = set()
    while True:
        try:
            # List objects in the bucket
            response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Delimiter='/')

            # Extract the common prefixes (directories)
            new_directories = {obj['Prefix'] for obj in response.get('CommonPrefixes', [])}

            # Filter out directories with the upload indicator
            new_directories = {d for d in new_directories if not d.startswith('uploading_')}

            if first_iteration:
                last_seen_objects = new_directories
                first_iteration = False
                continue

            # Find new directories
            added_directories = new_directories - last_seen_objects

            if added_directories:
                print("New directories found:")

                for directory in added_directories:
                    # List objects in the new directory
                    objects_in_directory = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=directory).get('Contents', [])

                    # Download each object in the new directory
                    for obj in objects_in_directory:
                        object_key = obj['Key']
                        local_file_path = os.path.join(LOCAL_DIRECTORY, object_key.split('/', 1)[1])
                        local_directory = os.path.dirname(local_file_path)
                        os.makedirs(local_directory, exist_ok=True)
                        # local_file_path = os.path.join(LOCAL_DIRECTORY, os.path.basename(object_key))

                        s3_client.download_file(BUCKET_NAME, object_key, local_file_path)
                        print(f"Downloaded: {object_key} to {local_file_path}")

                    ppo_trainer = PPOTrainer()
                    ppo_trainer.learn()

                    append_json_to_master(os.path.join(LOCAL_DIRECTORY, "agent_training_history.json"), LOCAL_MASTER_JSON_FILE)

                    delete_directory_and_contents(s3_client, directory)
                    # update trained model on bucket
                    local_file = os.path.join(LOCAL_DIRECTORY, "checkpoint.pt")
                    relative_key = os.path.relpath(local_file, LOCAL_DIRECTORY)
                    try:
                        s3_client.upload_file(local_file, BUCKET_NAME, relative_key)
                        print(f"File '{local_file}' uploaded to '{BUCKET_NAME}' successfully.")
                    except Exception as e:
                        print(f"Error uploading {local_file} to path {relative_key} at {BUCKET_NAME}: {str(e)}")

                    local_file = os.path.join(LOCAL_DIRECTORY, "history.json")
                    relative_key = os.path.relpath(local_file, LOCAL_DIRECTORY)
                    try:
                        s3_client.upload_file(local_file, BUCKET_NAME, relative_key)
                        print(f"File '{local_file}' uploaded to '{BUCKET_NAME}' successfully.")
                    except Exception as e:
                        print(f"Error uploading {local_file} to path {relative_key} at {BUCKET_NAME}: {str(e)}")

            # Update the last seen objects
            last_seen_objects = new_directories

        except Exception as e:
            print(f"Error: {str(e)}")

        time.sleep(1.0)
