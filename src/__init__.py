# Load the environment variables.
from dotenv import load_dotenv
load_dotenv()

import pathlib


# Create a base_dir variable which can be used if paths need to be specified.
BASE_DIR = pathlib.Path().parent.resolve()