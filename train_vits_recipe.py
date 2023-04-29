
import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig,CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
#import wandb 

DEFAULT_SAMPLE_RATE=16000
DEFAULT_DRIVE_FOLDER="tts-ai-results/"
os.makedirs(DEFAULT_DRIVE_FOLDER, exist_ok=True)


data_path="tts-ai-public-figure/"

characters_conf=CharactersConfig(
        pad="<PAD>",
        bos="<BOS>",
        eos="<EOS>",
        blank="<BLNK>",
        characters="abcdefghijklmnopqrstuvwxyzáéíñóú ",
        punctuations="!¡'(),-.:;¿?",
    )

#wandb.login() ##use open session to log in

project = "tts-petro-ai"
display_name = "VITS-es-1"

#wandb_log=wandb.init(project=project, name=display_name)


dataset_config = BaseDatasetConfig(
    formatter="ljspeech_custom",meta_file_train="metadata.txt", path=os.path.join(data_path,"tts-dataset/"))

# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.

config = VitsConfig(
    run_name=display_name,
    project_name=project,
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    save_checkpoints=True,
    save_n_checkpoints=2,
    save_best_after=1000,
    epochs=500,
    characters=characters_conf,
    text_cleaner="spanish_cleaners",
    use_phonemes=True,
    phoneme_language="es-es",
    phoneme_cache_path=os.path.join(DEFAULT_DRIVE_FOLDER,"phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=DEFAULT_DRIVE_FOLDER,
    datasets=[dataset_config],
#    dashboard_logger = 'wandb'
)

# ##AudioProcessor.init_from_config does not allow to set sample rate directly
config.audio['sample_rate']=DEFAULT_SAMPLE_RATE

## Audio processor
ap = AudioProcessor.init_from_config(config)
#=VitsAudioConfig.init_from_config(config)

# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)


#model = Tacotron2(config, ap, tokenizer, speaker_manager=None)
model=Vits(config, ap, tokenizer, speaker_manager=None)


## see TrainerArgs here (https://github.com/coqui-ai/Trainer/blob/main/trainer/trainer.py)
trainer = Trainer(
    TrainerArgs(gpu=0), ##default gpu device 0
    config, DEFAULT_DRIVE_FOLDER, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()
