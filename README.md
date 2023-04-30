# TTS-ai-public-figure

Training Text-to-speech using a public figure (Colombia's president Gustavo Petro) videos of public press conferences and speeches available online (mostly on youtube). This repo shows how to use the TTS library to train a custom voice model from a public figure voice. The following models were tested:

- VITS
- Tacotron2
- Glow TTS 

[VITS](https://arxiv.org/pdf/2106.06103.pdf) is one the most lightweight models we can use to train a voice model. The code provided in `tts_petro_ai_latest.ipynb` can be adapted to use any of the models listed above. Just bear in mind that more ram and vram are required.

The dataset was obtained manually, by pulling videos from youtube and partitioning them into chunks of 5 seconds each. Then, transcriptions were generated using [labelstudio.io](https://labelstud.io/).




## Setup environment

The TTS library had a small bug when creating this repo, for that reason i had to modify the formatters and cleanners module in order to use custom datasets and cleaners. You can install my forked version using the setup_env.sh. However, feel free to install the latest TTS release (`pip install TTS`).


```bash
chmod +x setup_env.sh
./setup_env.sh
```


## Training 

Run the config_data.py script, then use one of the recipes. Make sure to adjust your file paths (see variable `DEFAULT_DRIVE_FOLDER` in training scripts)


```bash
python3 config_data.py
python3 train_vits_recipe.py
```