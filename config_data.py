import numpy as np 
import pandas as pd 
import os
import logging
import re
import json
from pathlib import Path
import subprocess
import gdown
from pathlib import Path

logger=logging.getLogger("config_data")
logger.setLevel(logging.INFO)

##Download files and checkpoints 
OUT_PATH="/workspace/project/audio-processing/tts-ai-public-figure/training_results/"
FOLDER_NAME="tts-dataset"
Path(OUT_PATH).mkdir(exist_ok=True)


if __name__=="__main__":


    file_id = '1-uVNiBjN9d6GBZy3jubvgq7slyEgObyr'
    url = f'https://drive.google.com/uc?id='
    output = f'{OUT_PATH}best_model.pth'
    #gdown.download(f"{url}{file_id}", output, quiet=False)

    file_id = '1-Jot_x7bkWUE5SyAcKaaAAC-_T8DSEYe'
    url = f'https://drive.google.com/uc?id='
    output = f'{OUT_PATH}config.json'
    #gdown.download(f"{url}{file_id}", output, quiet=False)

    file_id = '157X6TUMSCLhcWxGIHkv-oQYlImnMMdkQ'
    url = f'https://drive.google.com/uc?id='
    output = '/workspace/project/audio-processing/tts-ai-public-figure/audios_labeled.zip' 
    gdown.download(f"{url}{file_id}", output, quiet=False)     #comment this line if you already have your own audio chunks

    logger.info("Data downloaded")


    subprocess.run(["unzip", "-q", "audios_labeled.zip"])     #comment this line if you already have your own audio chunks
    logger.info("Unzipped files")       #comment this line if you already have your own audio chunks

    subprocess.run(["mkdir",FOLDER_NAME])
    subprocess.run(["mv", "audios_labeled/chunks", "tts-dataset/wavs"])

    logger.info("Moved Data")

    subprocess.run(["mv", "audios_labeled/metadata.csv", "tts-dataset/"])

    logger.info("Moved Metadata")
