# TTS-AI-PUBLIC-FIGURE

Training Text-to-speech using a public figure (Colombia's president Gustavo Petro) videos of public press conferences and speeches available online (mostly on youtube). This repo shows how to use the TTS library to train a custom voice model from a public figure voice. The following models were tested:

- VITS
- Tacotron2
- Glow TTS 

[VITS](https://arxiv.org/pdf/2106.06103.pdf) is one the most lightweight models we can use to train a voice model. The code provided in `tts_petro_ai_latest.ipynb` can be adapted to use any of the models listed above. Just bear in mind that more ram and vram are required.

The dataset was obtained manually, by pulling videos from youtube and partitioning them into chunks of 5 seconds each. Then, transcriptions were generated using [labelstudio.io](https://labelstud.io/).





