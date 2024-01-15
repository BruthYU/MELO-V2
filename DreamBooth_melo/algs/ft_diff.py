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
import torch.nn.functional as F
from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.melo import MeloConfig, LoraLayer
from diffusers.optimization import get_scheduler
from tqdm import tqdm
from diffusers.training_utils import compute_snr
from diffusers import (
    DiffusionPipeline,
)
# from models import BertClassifier
LOG = logging.getLogger(__name__)

def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    global_step,
    instance_prompt
):

    LOG.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )


    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    text_encoder = accelerator.unwrap_model(text_encoder)
    unet = accelerator.unwrap_model(unet)

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
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)


    # Validation Prompt
    instance_prompt = instance_prompt.replace(args.instance_prompt, "").strip()
    pipeline_args = {"prompt": args.validation_prompt.format(instance_prompt)}

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

    for id, img in enumerate(images):
        img.save(f'{instance_prompt.replace(" ", "_")}_{global_step}_{id}.jpg')

    del pipeline
    torch.cuda.empty_cache()

class FT_DIFF(torch.nn.Module):
    def __init__(self, accelerator, tokenizer, noise_scheduler, vae, unet, text_encoder, config):
        super(FT_DIFF, self).__init__()
        self.config = config
        self.accelerator = accelerator
        self.block_index = 0
        self.tokenizer = tokenizer

        '''Get Basic Models
        '''
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.train_prepare()

    def train_prepare(self):
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        if self.config.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            self.optimizer_class = bnb.optim.AdamW8bit
        else:
            self.optimizer_class = torch.optim.AdamW

        if self.vae is not None:
            self.vae.to(self.accelerator.device, dtype=weight_dtype)

        if not self.config.train_text_encoder and self.text_encoder is not None:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        if self.vae is not None:
            self.vae.requires_grad_(False)

        self.noise_scheduler, self.vae, self.unet, self.text_encoder = \
            self.accelerator.prepare(self.noise_scheduler, self.vae, self.unet, self.text_encoder)

        self.weight_dtype = weight_dtype

    def tune(self, train_dataset, train_dataloader):
        params_to_optimize = (
            itertools.chain(self.unet.parameters(),
                            self.text_encoder.parameters()) if self.config.train_text_encoder else self.unet.parameters()
        )

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        cc = count_parameters(self.unet) + count_parameters(self.text_encoder)

        optimizer = self.optimizer_class(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.max_train_steps * self.accelerator.num_processes,
            num_cycles=self.config.lr_num_cycles,
            power=self.config.lr_power,
        )
        optimizer, lr_scheduler, train_dataloader = self.accelerator.prepare(optimizer, lr_scheduler, train_dataloader)
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)

        if overrode_max_train_steps:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.config.num_train_epochs = math.ceil(self.config.max_train_steps / num_update_steps_per_epoch)

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dreambooth")

            # Train!
            total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.gradient_accumulation_steps


            LOG.info(f"  Num examples = {len(train_dataset)}")
            LOG.info(f"  Num batches each epoch = {len(train_dataloader)}")
            LOG.info(f"  Recalculated Num Epochs = {self.config.num_train_epochs}")
            LOG.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
            LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            LOG.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            LOG.info(f"  Total optimization steps = {self.config.max_train_steps}")
            LOG.info(f"  Train text_encdoer = {self.config.train_text_encoder}")
            global_step = 0
            first_epoch = 0
            initial_global_step = 0

            progress_bar = tqdm(
                range(0, self.config.max_train_steps),
                initial=initial_global_step,
                desc="Steps",
                # Only show the progress bar once on each machine.
                disable=not self.accelerator.is_local_main_process,
            )

            for epoch in range(first_epoch, self.config.num_train_epochs):
                self.unet.train()
                if self.config.train_text_encoder:
                    self.text_encoder.train()
                for step, batch in enumerate(train_dataloader):
                    with self.accelerator.accumulate(self.unet):
                        pixel_values = batch["pixel_values"].to(dtype=self.weight_dtype)
                        if self.vae is not None:
                            # Convert images to latent space
                            model_input = self.vae.encode(batch["pixel_values"].to(dtype=self.weight_dtype)).latent_dist.sample()
                            model_input = model_input * self.vae.config.scaling_factor
                        else:
                            model_input = pixel_values

                        # Sample noise that we'll add to the model input
                        noise = torch.randn_like(model_input)

                        bsz, channels, height, width = model_input.shape
                        # Sample a random timestep for each image
                        timesteps = torch.randint(
                            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                        )
                        timesteps = timesteps.long()

                        # Add noise to the model input according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)

                        # Get the text embedding for conditioning
                        if self.config.pre_compute_text_embeddings:
                            encoder_hidden_states = batch["input_ids"]
                        else:
                            encoder_hidden_states = encode_prompt(
                                self.text_encoder,
                                batch["input_ids"],
                                batch["attention_mask"],
                                text_encoder_use_attention_mask=self.config.text_encoder_use_attention_mask,
                            )

                        if self.accelerator.unwrap_model(self.unet).config.in_channels == channels * 2:
                            noisy_model_input = torch.cat([noisy_model_input, noisy_model_input], dim=1)

                        if self.config.class_labels_conditioning == "timesteps":
                            class_labels = timesteps
                        else:
                            class_labels = None

                        # Predict the noise residual
                        model_pred = self.unet(noisy_model_input, timesteps, encoder_hidden_states).sample

                        if model_pred.shape[1] == 6:
                            model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                        # Get the target for loss depending on the prediction type
                        if self.noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif self.noise_scheduler.config.prediction_type == "v_prediction":
                            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
                        else:
                            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

                        if self.config.with_prior_preservation:
                            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                            target, target_prior = torch.chunk(target, 2, dim=0)
                            # Compute prior loss
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Compute instance loss
                        if self.config.snr_gamma is None:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                        else:
                            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                            # This is discussed in Section 4.2 of the same paper.
                            snr = compute_snr(self.noise_scheduler, timesteps)
                            base_weight = (
                                    torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                                        dim=1)[0] / snr
                            )

                            if self.noise_scheduler.config.prediction_type == "v_prediction":
                                # Velocity objective needs to be floored to an SNR weight of one.
                                mse_loss_weights = base_weight + 1
                            else:
                                # Epsilon and sample both use the same loss weights.
                                mse_loss_weights = base_weight
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                            loss = loss.mean()

                        if self.config.with_prior_preservation:
                            # Add the prior loss to the instance loss.
                            loss = loss + self.config.prior_loss_weight * prior_loss

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(self.unet.parameters(), self.text_encoder.parameters())
                                if self.config.train_text_encoder
                                else self.unet.parameters()
                            )
                            self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)

                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        if self.config.validation_prompt is not None and global_step % self.config.validation_steps == 0:
                            log_validation(
                                self.text_encoder,
                                self.tokenizer,
                                self.unet,
                                self.vae,
                                self.config,
                                self.accelerator,
                                self.weight_dtype,
                                global_step,
                                train_dataset.instance_prompt
                            )

                    logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)

                    if global_step >= self.config.max_train_steps:
                        break
            self.accelerator.free_memory()


    def save_pipeline(self):
        self.accelerator.wait_for_everyone()
        output_dir = self.config.output_dir
        if self.accelerator.is_main_process:
            unwrapped_unet = self.accelerator.unwrap_model(self.unet)
            unwrapped_unet.save_pretrained(
                os.path.join(output_dir, "unet"), state_dict=self.accelerator.get_state_dict(self.unet)
            )
            if self.config.train_text_encoder:
                unwrapped_text_encoder = self.accelerator.unwrap_model(self.text_encoder)
                unwrapped_text_encoder.save_pretrained(
                    os.path.join(output_dir, "text_encoder"),
                    state_dict=self.accelerator.get_state_dict(self.text_encoder)
                )
            self.accelerator.end_training()