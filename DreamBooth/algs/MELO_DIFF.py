from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os

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
    def __init__(self, tokenizer, scheduler, vae, unet, text_encoder, config):
        super(MELO_DIFF, self).__init__()
        self.config = config

        '''Get Basic Models
        '''
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder


        '''Melo Config
        '''
        r_num = config.model.num_block * config.model.num_rank_per_block
        self.unet_melo_config = MeloConfig(
            r= r_num,
            lora_alpha= r_num * 2,
            target_modules= list(config.model.UNET_TARGET_MODULES),
            lora_dropout= config.model.lora_dropout,
            fan_in_fan_out= config.model.fan_in_fan_out,
        )
        self.text_encoder_melo_config = MeloConfig(
            r= r_num,
            lora_alpha= r_num * 2,
            target_modules= list(config.model.TEXT_ENCODER_TARGET_MODULES),
            lora_dropout= config.model.lora_text_encoder_dropout,
            fan_in_fan_out= config.model.fan_in_fan_out,
        )
        self.log_dict = {}

        '''Apply MELO
        '''
        # self.original_model = model
        # self.model = model

        if not config.check_dir:
            self.text_encoder = get_peft_model(unet, self.text_encoder_melo_config)
            self.unet = get_peft_model(unet, self.unet_melo_config)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.unet_lora_list = self.named_lora_modules(self.unet)
        self.text_encoder_lora_list = self.named_lora_modules(self.text_encoder)


        '''Parameters to be optimized
        '''
        self.opt_params = self.optim_parameters()
        pass

    def named_lora_modules(self, model):
        module_list = [key for key, _ in model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(model.get_submodule(key), LoraLayer):
                lora_list.append(key)
        return lora_list
