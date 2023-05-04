import whisper
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

OUTPUT_FOLDER="/workspace/project/audio-processing/tts-ai-public-figure/chunks"
device="cuda:0" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(device)
#result = model.transcribe("audio.mp3")


results={}
cnt=0
for audio in tqdm(Path(OUTPUT_FOLDER).glob("*.wav"),desc="Transcribing audios"):
    try:
        prediction = model.transcribe(str(audio))
        results[str(audio).split("/")[-1]]=prediction['text'].strip()
    except Exception as e:
        print(f"Could not detect in {str(audio)} skipping. Error {e}")

data=pd.DataFrame.from_dict(results,orient="index")
data.to_csv("metadata_partial.csv",header=False)





