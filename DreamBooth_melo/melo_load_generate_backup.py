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
import json
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
from typing import *

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value



OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = logging.getLogger(__name__)



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
    identifier_list,
    subject_list,
):
    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    block_index = 0

    for sub, iden in zip(subject_list, identifier_list):

        unet.reset_dynamic_mapping([block_index])
        text_encoder.reset_dynamic_mapping([block_index])


        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=unwrap_peft(text_encoder),
            unet=unwrap_peft(unet),
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


        instance_name = sub.replace("_"," ")
        generality_prompt_list = prompt_for_generality_test(iden, instance_name)
        for idx, prompt in enumerate(generality_prompt_list):
            pipeline_args = {"prompt": prompt}

            # run inference
            LOG.info(
                f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                f" {prompt}."
            )
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

            folder = f"./generality/{sub}/prompt_id_{idx}"
            if not os.path.exists(folder):
                os.makedirs(folder)
            for i, img in enumerate(images):
                img.save(os.path.join(folder, f'{i}.jpg'))

        block_index += 1

    del pipeline
    torch.cuda.empty_cache()



def log_validation_locality(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    device,
    weight_dtype,
):
    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    unet.disable_adapter_layers()
    text_encoder.disable_adapter_layers()

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=unwrap_peft(text_encoder),
        unet=unwrap_peft(unet),
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

    locality_prompt_list = prompt_for_locality_test()
    for idx, prompt in enumerate(locality_prompt_list):
        pipeline_args = {"prompt": prompt}

        # run inference
        LOG.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {prompt}."
        )
        generator = None if args.locality_seed is None else torch.Generator(device=device).manual_seed(args.locality_seed)
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

        folder = f"./locality/prompt_id_{idx}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, img in enumerate(images):
            img.save(os.path.join(folder, f'{i}.jpg'))

    del pipeline
    torch.cuda.empty_cache()

def log_validation_reliability(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    device,
    weight_dtype,
    identifier_list,
    subject_list,
):
    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    block_index = 0

    for sub, iden in zip(subject_list, identifier_list):

        unet.reset_dynamic_mapping([block_index])
        text_encoder.reset_dynamic_mapping([block_index])
        block_index += 1
        instance_name = sub.replace("_", " ")
        reliability_prompt = f"a photo of {iden} {instance_name}"



        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=unwrap_peft(text_encoder),
            unet=unwrap_peft(unet),
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




        pipeline_args = {"prompt": reliability_prompt}

        # run inference
        LOG.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {reliability_prompt}."
        )
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

        folder = f"./reliability/{sub}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        for i, img in enumerate(images):
            img.save(os.path.join(folder, f'{i}.jpg'))

    del pipeline
    torch.cuda.empty_cache()

@hydra.main(config_path='config', config_name='config')
def run(config):
    LOG.info("*MELO* Load & Evaluation")
    base_dir = hydra.utils.get_original_cwd()
    device = torch.device('cuda')
    checkpoint_dir = os.path.join(base_dir,"eval/checkpoint/MELO/text-inversion-model")


    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(config.pretrained_model_name_or_path,
                                                                  config.revision)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision)
    text_encoder = PeftModel.from_pretrained(text_encoder,model_id=os.path.join(checkpoint_dir,"text_encoder"))


    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="vae", revision=config.revision)

    unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="unet", revision=config.revision)
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





    # Load the tokenizer
    if config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, revision=config.revision, use_fast=False)
    elif config.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=config.revision,
            use_fast=False,
        )

    with open(os.path.join(base_dir, "data", "data.json"), 'r') as f:
        data_info = json.load(f)
    subject_list = list(data_info.keys())
    identifier_list = np.load(os.path.join(base_dir, "data/rare_tokens/rare_tokens.npy"))[:len(subject_list)]

    log_validation(
        text_encoder,
        tokenizer,
        unet,
        vae,
        config,
        device,
        weight_dtype,
        identifier_list,
        subject_list
    )

    # log_validation_locality(
    #     text_encoder,
    #     tokenizer,
    #     unet,
    #     vae,
    #     config,
    #     device,
    #     weight_dtype,
    # )


    # log_validation_reliability(
    #     text_encoder,
    #     tokenizer,
    #     unet,
    #     vae,
    #     config,
    #     device,
    #     weight_dtype,
    #     identifier_list,
    #     subject_list,
    # )


    LOG.info("MELO-backened Dreambooth Evaluation Finishd")






if __name__ == '__main__':
    run()