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


def num2letters(text):
  n2l={'0':' cero','1':' uno','2':' dos','3':' tres','4':' cuatro','5':' cinco','6':' seis','7':' siete','8':' ocho','9':' nueve'}
  text=re.sub(r"(\d)", lambda x: n2l[str(x.group(0))], text)
  text=re.sub("\s\s+" , " ", text)
  return text



subprocess.run(["unzip", "-q", "audios_labeled.zip"])
logger.info("Unzipped files")

data=[json.loads(line) for line in open('manifest.json', 'r')]
data=[f"{i['audio_filepath'].split(chr(92))[-1].replace('.wav','')}|{num2letters(i['text'])}." for i in data]  # chr(92) is backslash
print(data[:5])


subprocess.run(["mkdir",FOLDER_NAME])
subprocess.run(["mv", "audio", "tts-dataset/wavs"])

logger.info("Moved Data")

file = open(f'{FOLDER_NAME}/metadata.txt','w')
for audio in data:
    file.write(audio+"\n")
file.close()