import logging
import random
from time import time
import os
import hydra
import struct
from omegaconf import OmegaConf,open_dict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import warnings
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
from packaging import version
import hashlib
import itertools
import math
import importlib
import shutil
import copy
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from dataset import *
from peft import MeloConfig, get_peft_model
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
LOG = logging.getLogger(__name__)

def check_config(config):
    base_dir = hydra.utils.get_original_cwd()
    config.instance_data_dir = os.path.join(base_dir, config.instance_data_dir)
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
    check_config(config)
    
    diff_config_keys = ['class_prompt', 'with_prior_preservation', 'prior_loss_weight', 'learning_rate']
    melo_config_keys = ['use_lora','UNET_TARGET_MODULES', 'TEXT_ENCODER_TARGET_MODULES']
    DIFF_CONFIG = dict(config)
    MELO_CONFIG = dict(config.model)
    for k in diff_config_keys:
        LOG.info(f'[-DIFF CONFIG-]  {k}: {DIFF_CONFIG[k]}')
    for k in melo_config_keys:
        LOG.info(f'[-MELO CONFIG-]  {k}: {MELO_CONFIG[k]}')

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

    pretrained_cache_dir = Path(hydra.utils.get_original_cwd(), config.pretrained_cache_dir)
    # Generate class images if prior preservation is enabled.
    if config.with_prior_preservation:
        class_images_dir = Path(config.base_dir, config.class_data_dir)

        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        # if not pretrained_cache_dir.exists():
        #     pretrained_cache_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if config.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif config.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif config.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.num_class_images - cur_class_images
            LOG.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(config.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=config.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    Load config
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


    alg_module = importlib.import_module(f'algs.{config.alg}')
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(tokenizer, noise_scheduler, vae, unet, text_encoder, config)


    pass


if __name__ == '__main__':
    run()
