import os
import TTS
from trainer import Trainer, TrainerArgs
from TTS.tts.models.vits import Vits, VitsAudioConfig,VitsArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig,CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.config import load_config
from config_data import OUT_PATH
import logging
logger=logging.getLogger("fine_tune_vits_recipe")
logger.setLevel(logging.INFO)


DEFAULT_SAMPLE_RATE=22000
DEFAULT_DRIVE_FOLDER=OUT_PATH
os.makedirs(DEFAULT_DRIVE_FOLDER, exist_ok=True)


vits_pre_conf=load_config(f"{OUT_PATH}config.json")
vits_pre_conf.audio.sample_rate=DEFAULT_SAMPLE_RATE


data_path="/workspace/project/audio-processing/tts-ai-public-figure/"

#wandb.login() ##use open session to log in

project = "tts-petro-ai"
display_name = "VITS-espeak-fine-tuning"


#wandb_log=wandb.init(project=project, name=display_name)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech",dataset_name="petro-tts",meta_file_train="metadata.csv", path=os.path.join(data_path,"tts-dataset/"),language="es"
)

vits_pre_conf.datasets=[dataset_config]
vits_pre_conf.output_path=DEFAULT_DRIVE_FOLDER



vits_pre_conf.test_sentences=[
        ['mi nombre es gustavo petro y esta es mi voz.'],
        ['en este país vamos a hacer las cosas mal.'],
        ['hola soy tu presidente petrosky.']]

##base config used from the original spanish vits model 
model_args_custom=VitsArgs(num_chars=129,use_language_embedding=False,inference_noise_scale=0.3,use_speaker_embedding=False,resblock_type_decoder='2',resblock_kernel_sizes_decoder=[3, 5, 7],
                          resblock_dilation_sizes_decoder=[[1, 2], [2, 6], [3, 12]],upsample_rates_decoder=[8, 8, 4],upsample_initial_channel_decoder=256,upsample_kernel_sizes_decoder=[16, 16, 8]
                           ,num_languages=0 )

vits_pre_conf.run_name=display_name
vits_pre_conf.project_name=project
vits_pre_conf.model_args=model_args_custom
vits_pre_conf.epochs=400
vits_pre_conf.save_step=1000
vits_pre_conf.num_loader_workers=2
vits_pre_conf.num_eval_loader_workers=2
vits_pre_conf.save_n_checkpoints=2
vits_pre_conf.output_path=DEFAULT_DRIVE_FOLDER
vits_pre_conf.max_audio_len=DEFAULT_SAMPLE_RATE*14
vits_pre_conf.lr=0.0001

## Audio processor
ap = AudioProcessor.init_from_config(vits_pre_conf)


# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(vits_pre_conf)
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_size=0.05,
)

for idx,v in enumerate(train_samples):
  train_samples[idx]['text']=train_samples[idx]['text'].replace("\n","")

for idx,v in enumerate(eval_samples):
  eval_samples[idx]['text']=eval_samples[idx]['text'].replace("\n","")


logger.info(f"Resulting data partitions:\t Train:{len(train_samples)} audios \t Val: {len(eval_samples)} audios")

model=Vits(config, ap, tokenizer, speaker_manager=None)


## see TrainerArgs here (https://github.com/coqui-ai/Trainer/blob/main/trainer/trainer.py)
trainer = Trainer(
    TrainerArgs(restore_path="/workspace/project/audio-processing/tts-ai-public-figure/training_results/best_model.pth",gpu="0"), ##default gpu device 0
    config, DEFAULT_DRIVE_FOLDER, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()