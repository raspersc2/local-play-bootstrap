from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()


S3_CONFIG = {
    "region_name": "ams3",
    "endpoint_url": "https://ams3.digitaloceanspaces.com",
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")
}
