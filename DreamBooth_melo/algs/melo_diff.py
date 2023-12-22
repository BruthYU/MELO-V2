from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os
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

        if self.vae is not None:
            self.vae.to(self.accelerator.device, dtype=weight_dtype)

        if not self.config.train_text_encoder and self.text_encoder is not None:
            self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)



    def edit(self, train_dataloader):
        pass



        


