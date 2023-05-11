# TTS-ai-public-figure

Training Text-to-speech using a public figure (Colombia's president Gustavo Petro) videos of public press conferences and speeches available online (mostly on youtube). This repo shows how to use the TTS library to train a custom voice model from a public figure voice. 
 

## Model
The model used was a fine-tuned version of [VITS](https://arxiv.org/pdf/2106.06103.pdf). Given its architecture (parallel end to-end TTS) is one the most straightforward options we can use to train a voice model (Vocoder tuning is not required). 


## Dataset 

The dataset was obtained manually, by pulling videos from youtube and partitioning them into chunks obtained from a silence threshold (`pydub.split_on_silence()`). Then, transcriptions were generated using OPEN AI's Whisper. Some manual supervision is required as this can return transcriptions in a different language. [NoiseReduce](https://github.com/timsainb/noisereduce/blob/master/noisereduce/noisereduce.py) is used to reduce noise before getting transcriptions. See `create_audio_partitions.py` for more details. 


## Training


### Setup environment


- [x] A GPU is needed for training. While building this repo, Kaggle notebooks and a local instance with a single RTX 3080 12 Gb | 16 Gb ram were used for experimentation.

- [x] Use the `setup_env.sh` script to install requirements. If you have issues using your custom dataset cleaner or formatter, you can follow the steps i used in a forked version of the TTS repo (modify the cleaners.py and formatters.py). However, feel free to install the latest TTS release (`pip install TTS`).


```bash
chmod +x setup_env.sh
./setup_env.sh
```

### Preproc & Training script 

Once you have collected the videos in mp4 format, save them in a folder. Modify the `VIDEOS_PATH` variable in create_audio_partitions.py to point to the videos folder. Run the scripts create_audio_partitions.py and config_data.py as well, then use one of the recipes (this can be obtained from the TTS repo directly) or use the fine-tuned script( this will download my weights from google drive as well as the configuration file required). Make sure to adjust your file paths (see variable `DEFAULT_DRIVE_FOLDER` in training scripts).


```bash
python3 create_audio_partitions.py
python3 config_data.py
python3 translate_audios.py 
python3 "<fine_tune|train>"_vits_recipe.py ## fine-tuning is recommended as training from scratch will require a lot of data and iterations to get decent outputs.
```

An instance of tensorboard can be loaded for monitoring, the training script will provide the logs_dir. In a new terminal run the following.

```bash
tensorboard --logdir=<YOUR_LOG_DIR_PATH>
```

## Inference

Use TTS built-in script for inference or its API. For better results it is recommended to use audio-denoising tools like NoiseReduce if the wav files seem to be noisy.

```bash

tts --text "<YOUR TEXT HERE>" \
      --model_path "<PATH_TO_YOUR_CHECKPOINT>/best_model.pth" \
      --config_path "<PATH_TO_YOUR_CHECKPOINT>/config.json" \
      --out_path audio_test.wav \
      --use_cuda "0" ##only if you have gpu-enabled
```


Examples of the outputs can be found in the results/ folder


https://github.com/statscol/tts-ai-public-figure/assets/47220701/d792a9c2-196f-4c5b-9408-a19a5e6d907c



## Demo

A demo app can be found in [this](https://huggingface.co/spaces/jhonparra18/petro-tts-app) HuggingFace Space. This will be updated with the latest checkpoint.
