
import os
import TTS
from trainer import Trainer, TrainerArgs
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig,CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
#import wandb 
import logging
logger=logging.getLogger("train_vits_recipe")
logger.setLevel(logging.INFO)

DEFAULT_SAMPLE_RATE=22000
DEFAULT_DRIVE_FOLDER="/workspace/project/audio-processing/tts-ai-public-figure/training_results/"
os.makedirs(DEFAULT_DRIVE_FOLDER, exist_ok=True)

audio_conf=VitsAudioConfig(fft_size=1024, sample_rate=DEFAULT_SAMPLE_RATE, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None)
data_path="/workspace/project/audio-processing/tts-ai-public-figure/"

characters_conf=CharactersConfig(
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters="abcdefghijklmnopqrstuvwxyzáéíñóú ",
        punctuations="!¡'(),-.:;¿?",
    )


#wandb.login() ##use open session to log in

project = "tts-petro-ai"
display_name = "VITS-es-1"

#wandb_log=wandb.init(project=project, name=display_name)


dataset_config = BaseDatasetConfig(
    formatter="ljspeech",meta_file_train="metadata.csv", path=os.path.join(data_path,"tts-dataset/"))

# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.

config = VitsConfig(
    run_name=display_name,
    project_name=project,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    save_checkpoints=True,
    save_n_checkpoints=2,
    save_best_after=500,
    save_step=500,
    epochs=1000,
    characters=characters_conf,
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language="es-es",
    phoneme_cache_path=os.path.join(DEFAULT_DRIVE_FOLDER,"phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=False,
    mixed_precision=True,
    output_path=DEFAULT_DRIVE_FOLDER,
    datasets=[dataset_config],
#    dashboard_logger = 'wandb'
)

## Audio processor
ap = AudioProcessor.init_from_config(config)
#=VitsAudioConfig.init_from_config(config)

#config.characters.print_logs()

# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)


logger.info(f"Resulting data partitions:\t Train:{len(train_samples)} audios \t Val: {len(eval_samples)} audios")

model=Vits(config, ap, tokenizer, speaker_manager=None)


## see TrainerArgs here (https://github.com/coqui-ai/Trainer/blob/main/trainer/trainer.py)
trainer = Trainer(
    TrainerArgs(gpu=0), ##default gpu device 0
    config, DEFAULT_DRIVE_FOLDER, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()
