import os
import transformers
import warnings
import json
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from trainer.ft_trainer import *
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
import numpy as np
LOG = logging.getLogger(__name__)


def check_config(config):
    base_dir = hydra.utils.get_original_cwd()
    config.class_data_dir = os.path.join(base_dir, config.class_data_dir)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.local_rank:
        config.local_rank = env_local_rank
    if config.with_prior_preservation:
        if config.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if config.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if config.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if config.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")
    if config.train_text_encoder and config.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

@hydra.main(config_path='config', config_name='config')
def run(config):
    LOG.info("*Fine-Tuning* Dreambooth")
    check_config(config)
    base_dir = hydra.utils.get_original_cwd()
    with open_dict(config):
        config.base_dir = base_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision
    )

    # Gradient Accumulation is not supported for multi_gpu setting
    if config.train_text_encoder and config.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if accelerator.is_local_main_process:
        transformers.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    '''
    Load Model
    '''
    text_encoder_cls = import_model_class_from_model_name_or_path(config.pretrained_model_name_or_path,
                                                                  config.revision)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision)
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision)

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision)

    '''
    Load tokenizer
    '''
    if config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, revision=config.revision, use_fast=False)
    elif config.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=config.revision,
            use_fast=False,
        )

    '''
    Algorithm Initialization
    '''
    alg_module = importlib.import_module(f'algs.{config.alg}')
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(accelerator, tokenizer, noise_scheduler, vae, unet, text_encoder, config)

    '''
    data_info
    '''
    with open(os.path.join(base_dir, "data","data.json"), 'r') as f:
        data_info = json.load(f)
    subject_list = list(data_info.keys())
    identifier_list = np.load(os.path.join(base_dir,'data/rare_tokens/rare_tokens.npy'))[:len(subject_list)]

    '''
    Trainer
    '''
    trainer = ft_trainer(config, alg, accelerator, tokenizer, None, data_info, subject_list, identifier_list)
    trainer.prepare_dataset()
    torch.cuda.empty_cache()
    trainer.run_fine_tune()





if __name__ == '__main__':
    run()