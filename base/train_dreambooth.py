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
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'


def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10 ** digits)

    return uuid.uuid_value



OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = get_logger(__name__)

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

def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    accelerator,
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
        text_encoder = accelerator.unwrap_model(text_encoder)

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
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
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.pre_compute_text_embeddings:
        pipeline_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        pipeline_args = {"prompt": args.validation_prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
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
    args = config.args
    check_config(config)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.args.gradient_accumulation_steps,
        mixed_precision=config.args.mixed_precision
    )

    # Gradient Accumulation is not supported for multi_gpu setting
    if config.args.train_text_encoder and config.args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
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
    if config.args.seed is not None:
        set_seed(config.args.seed)

    pretrained_cache_dir = Path(hydra.utils.get_original_cwd(), config.args.pretrained_cache_dir)
    # Generate class images if prior preservation is enabled.
    if config.args.with_prior_preservation:
        class_images_dir = Path(hydra.utils.get_original_cwd(), config.args.class_data_dir)

        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        # if not pretrained_cache_dir.exists():
        #     pretrained_cache_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < config.args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if config.args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif config.args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif config.args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                config.args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=config.args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = config.args.num_class_images - cur_class_images
            LOG.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(config.args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=config.args.sample_batch_size)

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







    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(config.args.pretrained_model_name_or_path,
                                                                  config.args.revision)
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(config.args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        config.args.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.args.revision)
    vae = AutoencoderKL.from_pretrained(
        config.args.pretrained_model_name_or_path, subfolder="vae", revision=config.args.revision)

    unet = UNet2DConditionModel.from_pretrained(
        config.args.pretrained_model_name_or_path, subfolder="unet", revision=config.args.revision)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if vae is not None:
        vae.requires_grad_(False)

    if not config.args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if config.args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                LOG.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if config.args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

        # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if config.args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.args.scale_lr:
        config.args.learning_rate = (
                config.args.learning_rate * config.args.gradient_accumulation_steps * config.args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if config.args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = (
        itertools.chain(unet.parameters(),
                        text_encoder.parameters()) if config.args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=config.args.learning_rate,
        betas=(config.args.adam_beta1, config.args.adam_beta2),
        weight_decay=config.args.adam_weight_decay,
        eps=config.args.adam_epsilon,
    )

    pre_computed_encoder_hidden_states = None
    pre_computed_class_prompt_encoder_hidden_states = None
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

    train_dataset = DreamBoothDataset(
        instance_data_root=config.args.instance_data_dir,
        instance_prompt=config.args.instance_prompt,
        class_data_root=config.args.class_data_dir if config.args.with_prior_preservation else None,
        class_prompt=config.args.class_prompt,
        class_num=config.args.num_class_images,
        tokenizer=tokenizer,
        size=config.args.resolution,
        center_crop=config.args.center_crop,
        encoder_hidden_states=pre_computed_encoder_hidden_states,
        class_prompt_encoder_hidden_states=pre_computed_class_prompt_encoder_hidden_states,
        tokenizer_max_length=config.args.tokenizer_max_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, config.args.with_prior_preservation),
        num_workers=config.args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.args.gradient_accumulation_steps)

    if config.args.max_train_steps is None:
        config.args.max_train_steps = config.args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.args.max_train_steps * accelerator.num_processes,
        num_cycles=config.args.lr_num_cycles,
        power=config.args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if config.args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    if not config.args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.args.max_train_steps = config.args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.args.num_train_epochs = math.ceil(config.args.max_train_steps / num_update_steps_per_epoch)


    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = config.args.train_batch_size * accelerator.num_processes * config.args.gradient_accumulation_steps

    LOG.info("***** Running training *****")
    LOG.info(f"  Num examples = {len(train_dataset)}")
    LOG.info(f"  Num batches each epoch = {len(train_dataloader)}")
    LOG.info(f"  Num Epochs = {config.args.num_train_epochs}")
    LOG.info(f"  Instantaneous batch size per device = {config.args.train_batch_size}")
    LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    LOG.info(f"  Gradient Accumulation steps = {config.args.gradient_accumulation_steps}")
    LOG.info(f"  Total optimization steps = {config.args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, config.args.num_train_epochs):
        unet.train()
        if config.args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                if vae is not None:
                    # Convert images to latent space
                    model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values

                # Sample noise that we'll add to the model input
                if config.args.offset_noise:
                    noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                        model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                    )
                else:
                    noise = torch.randn_like(model_input)
                bsz, channels, height, width = model_input.shape
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                )
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                if config.args.pre_compute_text_embeddings:
                    encoder_hidden_states = batch["input_ids"]
                else:
                    encoder_hidden_states = encode_prompt(
                        text_encoder,
                        batch["input_ids"],
                        batch["attention_mask"],
                        text_encoder_use_attention_mask=config.args.text_encoder_use_attention_mask,
                    )

                if accelerator.unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                if config.args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                ).sample

                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if config.args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Compute instance loss
                if config.args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    base_weight = (
                        torch.stack([snr, config.args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective needs to be floored to an SNR weight of one.
                        mse_loss_weights = base_weight + 1
                    else:
                        # Epsilon and sample both use the same loss weights.
                        mse_loss_weights = base_weight
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                if config.args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + config.args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if config.args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, config.args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=config.args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % config.args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if config.args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(config.args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= config.args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                LOG.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                LOG.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        LOG.info(f"Saved state to {save_path}")

                    images = []

                    if config.args.validation_prompt is not None and global_step % config.args.validation_steps == 0:
                        log_validation(
                            text_encoder,
                            tokenizer,
                            unet,
                            vae,
                            config.args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            validation_prompt_encoder_hidden_states,
                            validation_prompt_negative_prompt_embeds,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config.args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if config.args.lora.use_lora:
            unwarpped_unet = accelerator.unwrap_model(unet)
            unwarpped_unet.save_pretrained(
                os.path.join(config.args.output_dir, "unet"), state_dict=accelerator.get_state_dict(unet)
            )
            if config.args.train_text_encoder:
                unwarpped_text_encoder = accelerator.unwrap_model(text_encoder)
                unwarpped_text_encoder.save_pretrained(
                    os.path.join(config.args.output_dir, "text_encoder"),
                    state_dict=accelerator.get_state_dict(text_encoder),
                )
        else:
            pipeline_args = {}
            if text_encoder is not None:
                pipeline_args["text_encoder"] = accelerator.unwrap_model(text_encoder)
            if config.args.skip_save_text_encoder:
                pipeline_args["text_encoder"] = None
            pipeline = DiffusionPipeline.from_pretrained(
                config.args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                revision=config.args.revision,
                **pipeline_args,
            )
            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_args = {}
            if "variance_type" in pipeline.scheduler.config:
                variance_type = pipeline.scheduler.config.variance_type
                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"
                scheduler_args["variance_type"] = variance_type
            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
            pipeline.save_pretrained(config.args.output_dir)

        accelerator.end_training()




if __name__ == '__main__':
    run()