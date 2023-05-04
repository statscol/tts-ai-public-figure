# TTS-ai-public-figure

Training Text-to-speech using a public figure (Colombia's president Gustavo Petro) videos of public press conferences and speeches available online (mostly on youtube). This repo shows how to use the TTS library to train a custom voice model from a public figure voice. The following models were tested:

- VITS
- Glow TTS 

[VITS](https://arxiv.org/pdf/2106.06103.pdf), given its architecture (parallel end to-end TTS) is one the most straightforward options we can use to train a voice model. The code provided in `tts_petro_ai_latest.ipynb` can be adapted to use any of the models listed above. Just bear in mind that more ram and vram are required.

The dataset was obtained manually, by pulling videos from youtube and partitioning them into chunks obtained from a silence threshold (`pydub.split_on_silence()`). Then, transcriptions were generated using OPEN AI's Whisper. Some manual supervision is required as this can return transcriptions in a different language. [NoiseReduce](https://github.com/timsainb/noisereduce/blob/master/noisereduce/noisereduce.py) is used to reduce noise before getting transcriptions. See `create_audio_partitions.py` for more details. 


## Setup environment

The TTS library had a small bug when creating this repo, for that reason i had to modify the formatters and cleanners module in order to use custom datasets and cleaners. You can install my forked version using the setup_env.sh. However, feel free to install the latest TTS release (`pip install TTS`).


```bash
chmod +x setup_env.sh
./setup_env.sh
```


## Training 

Once you have collected videos from youtube in mp4 format, save them in a folder. Modify the `VIDEOS_PATH` variable in create_audio_partitions.py to point to the videos folder. Run the scripts create_audio_partitions.py and config_data.py as well, then use one of the recipes (this can be obtained from the TTS repo directly). Make sure to adjust your file paths (see variable `DEFAULT_DRIVE_FOLDER` in training scripts).


```bash
python3 create_audio_partitions.py
python3 config_data.py
python3 train_vits_recipe.py
```

## Inference

Use TTS build-in script for inference. It is recommended to use audio-denoising tools like NoiseReduce if the wav files seem to be noisy.

```bash

tts --text "<YOUR TEXT HERE>" \
      --model_path "<PATH_TO_YOUR_CHECKPOINT>/best_model.pth" \
      --config_path "<PATH_TO_YOUR_CHECKPOINT>/config.json" \
      --out_path audio_test.wav \
      --use_cuda "0" ##only if you have gpu-enabled
```

## Demo

WIP
