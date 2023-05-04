from pydub import AudioSegment
from pydub.utils import make_chunks 
from pydub.silence import split_on_silence
from pathlib import Path
from typing import List
from tqdm import tqdm
from scipy.io import wavfile
import noisereduce as nr
import numpy as np


DEFAULT_SAMPLE_RATE=22000
OUTPUT_FOLDER="/workspace/project/audio-processing/tts-ai-public-figure/chunks"


def std_pydub_audio(audio):
    ##taken from pydub repo
    samples = audio.get_array_of_samples()
    audio_arr = np.array(samples).T.astype(np.float32)
    audio_arr /= np.iinfo(samples.typecode).max
    return audio_arr


def reduce_noise(audio,sr,filepath):
    reduced_noise = nr.reduce_noise(y=audio, sr=sr)
    wavfile.write(filepath, sr, reduced_noise)
    return True

def create_chunks_mp4(file_path:str,save_to_disk:bool=False)->List[AudioSegment]:
    audios = AudioSegment.from_file(file_path,'mp4')
    audios = audios.set_frame_rate(DEFAULT_SAMPLE_RATE) #conver to 22000Hz
    audios = audios.set_channels(1) ##conver to mono
    audios=split_on_silence(audios,min_silence_len=500, silence_thresh=-40)
    #audios=make_chunks(audios,5000) ##chunks of 5 seconds
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    if save_to_disk:
        for idx,audio in tqdm(enumerate(audios),desc="Saving Audios"):
            file_name=file_path.split("/")[-1].replace('.mp4','.wav')
            file_name=f"{OUTPUT_FOLDER}/{idx}{file_name}"
            #audio.export(f"{OUTPUT_FOLDER}/{idx}{file_name}", format="wav")
            audio=std_pydub_audio(audio)
            #perform noise reduction
            reduce_noise(audio,DEFAULT_SAMPLE_RATE,file_name)
    ## convertion to float32 # compatibility with librosa output
    return audios


if __name__=="__main__":

    for vid in Path("/workspace/project/videos/").glob("*.mp4"):
        create_chunks_mp4(str(vid),save_to_disk=True)



