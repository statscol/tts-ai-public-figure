from pathlib import Path
import pandas as pd
import numpy as np
import re

def num2letters(text):
  n2l={'0':' cero','1':' uno','2':' dos','3':' tres','4':' cuatro','5':' cinco','6':' seis','7':' siete','8':' ocho','9':' nueve'}
  text=re.sub(r"(\d)", lambda x: n2l[str(x.group(0))], text)
  text=re.sub("\s\s+" , " ", text)
  return text

OUTPUT_FOLDER="/workspace/project/audio-processing/tts-ai-public-figure/chunks"
audios_partial=pd.read_csv("metadata_partial.csv",header=None)
audios_partial.columns=['filename','text']
audios_partial=audios_partial[(audios_partial.text.apply(lambda d: len(str(d).strip()))>0)]
audios_partial=audios_partial[audios_partial.text.apply(lambda d: d is not np.nan)]
audios_partial['text']=audios_partial.text.apply(lambda d: num2letters(str(d)))
audios_partial['filename']=audios_partial.filename.apply(lambda d: d.replace(".wav",""))
audios_partial['text_dup']=audios_partial['text']
audios_partial.to_csv("metadata.csv",header=False,sep="|",index=False)

for audio in Path(OUTPUT_FOLDER).glob("*.wav"):
  if audio.name not in audios_partial.filename.values.tolist():
    ##remove files that were discarded
    audio.unlink(missing_ok=True)



