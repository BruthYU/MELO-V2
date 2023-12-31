from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os
import math
import itertools
from torch.nn import Parameter
import importlib
from utils import *
from PIL import Image
import warnings
import hydra
from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
# from models import BertClassifier
LOG = logging.getLogger(__name__)


os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
identifier_list = ["sks", "Tom's", "Jackie's", "Cunningham‘s", "Lang's"]
subject_list = ["rc_car", "shiny_sneaker", "cat", "vase", "pink_sunglasses"]


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
):

    LOG.info(
        f"[Running validation]"
    )

    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    for idx, (identifier, subject) in enumerate(zip(identifier_list, subject_list)):

        unet.reset_dynamic_mapping([idx])
        text_encoder.reset_dynamic_mapping([idx])

        # create pipeline (note: lora_mapping in unet and text_encoder)
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

        subject = subject.replace("_", " ")
        instance_prompt = " ".join([identifier, subject])
        LOG.info(f"Generating {args.num_validation_images} images with prompt: "
                 f"{args.validation_prompt.format(instance_prompt)}.")

        pipeline_args = {"prompt": args.validation_prompt.format(instance_prompt)}

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

        for img_idx, img in enumerate(images):
            img.save(f'eval_{instance_prompt.replace(" ","_")}_{img_idx}.jpg')

    del pipeline
    torch.cuda.empty_cache()


@hydra.main(config_path='config', config_name='config')
def run(config):
    base_dir = hydra.utils.get_original_cwd()
    device = torch.device('cuda')
    checkpoint_dir = os.path.join(base_dir,"outputs/2023-12-28_09-39-20/text-inversion-model")
    check_config(config)


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



    validation_prompt_encoder_hidden_states = None
    validation_prompt_negative_prompt_embeds = None

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

    log_validation(
        text_encoder,
        tokenizer,
        unet,
        vae,
        config,
        device,
        weight_dtype,
    )
    LOG.info("MELO-backened Dreambooth Evaluation Finished")

if __name__ == '__main__':
    run()
    
