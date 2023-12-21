import logging
from time import time
import os
import hydra
import logging
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
# from accelerate.logging import get_logger
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
from peft import LoraConfig, get_peft_model, PeftModel
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
from typing import *

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value



OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = logging.getLogger(__name__)

def check_config(config):
    base_dir = hydra.utils.get_original_cwd()
    config.args.instance_data_dir = os.path.join(base_dir, config.args.instance_data_dir)
    config.args.class_data_dir = os.path.join(base_dir, config.args.class_data_dir)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != config.args.local_rank:
        config.args.local_rank = env_local_rank
    if config.args.with_prior_preservation:
        if config.args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if config.args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if config.args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if config.args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")
    if config.args.train_text_encoder and config.args.pre_compute_text_embeddings:
        raise ValueError("`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`")
# Dataset and DataLoaders creation:

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

def unwrap_peft(input_model):
    if input_model.__class__.__name__ == "PeftModel":
        input_model = input_model.base_model.model
    return input_model

def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    device,
    weight_dtype,
    global_step,
    prompt_embeds,
    negative_prompt_embeds,
):

    LOG.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )


    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    if text_encoder is not None:
        text_encoder = unwrap_peft(text_encoder)

    if unet is not None:
        unet = unwrap_peft(unet)


    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        revision=args.revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
        **pipeline_args,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.pre_compute_text_embeddings:
        pipeline_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        pipeline_args = {"prompt": args.validation_prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=device).manual_seed(args.seed)
    images = []
    if args.validation_images is None:
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
            images.append(image)
    else:
        for image in args.validation_images:
            image = Image.open(image)
            image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
            images.append(image)

    for id,img in enumerate(images):
        img.save(f'validation_{global_step}_{id}.jpg')

    del pipeline
    torch.cuda.empty_cache()





@hydra.main(config_path='config', config_name='config')
def run(config):
    base_dir = hydra.utils.get_original_cwd()
    device = torch.device('cuda')
    checkpoint_dir = os.path.join(base_dir,"outputs/2023-12-04_10-38-20/text-inversion-model")
    check_config(config)


    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(config.args.pretrained_model_name_or_path,
                                                                  config.args.revision)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(config.args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        config.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.args.revision)
    text_encoder = PeftModel.from_pretrained(text_encoder,model_id=os.path.join(checkpoint_dir,"text_encoder"))


    vae = AutoencoderKL.from_pretrained(
        config.args.pretrained_model_name_or_path, subfolder="vae", revision=config.args.revision)

    unet = UNet2DConditionModel.from_pretrained(
        config.args.pretrained_model_name_or_path, subfolder="unet", revision=config.args.revision)
    unet = PeftModel.from_pretrained(unet,model_id=os.path.join(checkpoint_dir,"unet"))

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32


    # Move vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(device, dtype=weight_dtype)

    if text_encoder is not None:
        text_encoder.to(device, dtype=weight_dtype)

    if unet is not None:
        unet.to(device, dtype=weight_dtype)



    validation_prompt_encoder_hidden_states = None
    validation_prompt_negative_prompt_embeds = None

    # Load the tokenizer
    if config.args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(config.args.tokenizer_name, revision=config.args.revision, use_fast=False)
    elif config.args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=config.args.revision,
            use_fast=False,
        )

    log_validation(
        text_encoder,
        tokenizer,
        unet,
        vae,
        config.args,
        device,
        weight_dtype,
        "eval",
        validation_prompt_encoder_hidden_states,
        validation_prompt_negative_prompt_embeds,
    )
    LOG.info("Peft-backened Dreambooth Evaluation Finishd")






if __name__ == '__main__':
    run()