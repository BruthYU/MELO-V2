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

from utils import *

from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.melo import MeloConfig, LoraLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
# from models import BertClassifier
LOG = logging.getLogger(__name__)

class MELO_DIFF(torch.nn.Module):
    def __init__(self, accelerator, scheduler, vae, unet, text_encoder, config):
        super(MELO_DIFF, self).__init__()
        self.config = config
        self.accelerator = accelerator

        '''Get Basic Models
        '''
        self.scheduler = scheduler
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder



        '''Melo Config
        '''
        r_num = config.model.num_block * config.model.num_rank_per_block.unet
        self.unet_melo_config = MeloConfig(
            r = r_num,
            lora_alpha= r_num * 2,
            target_modules= list(config.model.UNET_TARGET_MODULES),
            lora_dropout= config.model.lora_dropout,
            fan_in_fan_out= config.model.fan_in_fan_out,
            num_rank_per_block = config.model.num_rank_per_block.unet
        )

        r_num = config.model.num_block * config.model.num_rank_per_block.text_encoder
        self.text_encoder_melo_config = MeloConfig(
            r= r_num,
            lora_alpha= r_num * 2,
            target_modules= list(config.model.TEXT_ENCODER_TARGET_MODULES),
            lora_dropout= config.model.lora_text_encoder_dropout,
            fan_in_fan_out= config.model.fan_in_fan_out,
            num_rank_per_block=config.model.num_rank_per_block.text_encoder
        )
        self.log_dict = {}

        '''Apply MELO
        '''
        # self.original_model = model
        # self.model = model

        if not config.check_dir:
            self.text_encoder = get_peft_model(text_encoder, self.text_encoder_melo_config)
            self.unet = get_peft_model(unet, self.unet_melo_config)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.unet_lora_list = self.named_lora_modules(self.unet)
        self.text_encoder_lora_list = self.named_lora_modules(self.text_encoder)

        '''
        Only LoRA modules are marked as trainable (mark_only_lora_as_trainable)
        '''
        self.train_prepare()


    def named_lora_modules(self, model):
        module_list = [key for key, _ in model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(model.get_submodule(key), LoraLayer):
                lora_list.append(key)
        return lora_list

    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()
        self.model.base_model.disable_grace_layer()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()
        self.model.base_model.enable_grace_layer()
        
    def train_prepare(self):
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.config.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.params_to_optimize = (
            itertools.chain(self.unet.parameters(),
                            self.text_encoder.parameters()) if self.config.train_text_encoder else self.unet.parameters()
        )

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
        
        self.scheduler,self.vae, self.unet,self.text_encoder = \
            self.accelerator.prepare(self.scheduler,self.vae, self.unet,self.text_encoder)
        



    def edit(self, train_dataset, train_dataloader):
        optimizer = self.optimizer_class(
            self.params_to_optimize,
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
        optimizer, lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)
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

            LOG.info("***** Running training *****")
            LOG.info(f"  Num examples = {len(train_dataset)}")
            LOG.info(f"  Num batches each epoch = {len(train_dataloader)}")
            LOG.info(f"  Num Epochs = {self.config.num_train_epochs}")
            LOG.info(f"  Instantaneous batch size per device = {self.config.train_batch_size}")
            LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            LOG.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            LOG.info(f"  Total optimization steps = {self.config.max_train_steps}")
            global_step = 0
            first_epoch = 0
            initial_global_step = 0



        
        



        


