import numpy as np 
import pandas as pd 
import os
import logging
import re
import json
from pathlib import Path
import subprocess


logger=logging.getLogger("config_data")
logger.setLevel(logging.INFO)

FOLDER_NAME="tts-dataset"


subprocess.run(["unzip", "-q", "audios_labeled.zip"])
logger.info("Unzipped files")

subprocess.run(["mkdir",FOLDER_NAME])
subprocess.run(["mv", "chunks", "tts-dataset/wavs"])

logger.info("Moved Data")

subprocess.run(["mv", "metadata.csv", "tts-dataset/"])

logger.info("Moved Metadata")
