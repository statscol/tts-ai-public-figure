from pydub import AudioSegment
from pydub.utils import make_chunks 
from pydub.silence import split_on_silence
from pathlib import Path
from typing import List
from tqdm import tqdm

OUTPUT_FOLDER="chunks"
def create_chunks_mp4(file_path:str,save_to_disk:bool=False)->List[AudioSegment]:
    audios = AudioSegment.from_file(file_path,'mp4')
    audios = audios.set_frame_rate(16000) #conver to 16000Hz
    audios = audios.set_channels(1) ##conver to mono
    #audios=split_on_silence(audios,min_silence_len=100, silence_thresh=-5)
    audios=make_chunks(audios,5000) ##chunks of 5 seconds
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    if save_to_disk:
        for idx,audio in tqdm(enumerate(audios),desc="Saving Audios"):
            audio.export(f"{OUTPUT_FOLDER}/{idx}{file_path.replace('.mp4','.wav')}", format="wav")
    ## convertion to float32 # compatibility with librosa output
    return audios


if __name__=="__main__":

    for vid in Path(".").glob("*.mp4"):
        create_chunks_mp4(str(vid),save_to_disk=True)


