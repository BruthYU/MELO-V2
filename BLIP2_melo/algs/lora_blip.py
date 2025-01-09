from typing import List
from omegaconf import OmegaConf
import torch
import copy
import transformers
import logging
import os
from torchvision import transforms
from torch.nn import Parameter
from clip_model import *
import itertools

from utils import *

from peft import (
    PeftModel,
    prepare_model_for_int8_training,
    MeloConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.melo import LoraLayer

# from models import BertClassifier
LOG = logging.getLogger(__name__)

def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)



class LORA_BLIP(torch.nn.Module):
    def __init__(self, model, config, scale=None):
        super(LORA_BLIP, self).__init__()
        self.config = config

        '''Apply_melo
        '''
        r_num = config.melo.num_block * config.melo.num_rank_per_block
        self.lora_config = MeloConfig(
            r=r_num,
            lora_alpha=r_num,
            target_modules=list(config.model.target_modules),
            lora_dropout=config.lora.lora_dropout,
            fan_in_fan_out=config.model.fan_in_fan_out,
            num_rank_per_block=config.melo.num_rank_per_block

        )
        self.log_dict = {}

        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config)
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_lora_modules()
        self.outputs = {}

        '''Load Tokenizer
        '''



    def save_lora_weights(self, lora_dir):
        self.model.save_pretrained(lora_dir + "/lora_checkpoint")

    def named_lora_modules(self):
        module_list = [key for key, _ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            if isinstance(self.model.get_submodule(key), LoraLayer):
                lora_list.append(key)
        return lora_list

    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()

    def set_lora_mapping(self, lora_block_mapping):
        self.model.reset_dynamic_mapping(lora_block_mapping)

    def edit(self, batch, batch_index):
        # MELO_V2 could automatically identify lora parameters to be optimized
        params_to_optimize = (itertools.chain(self.model.parameters()))
        optimizer = torch.optim.Adam(params_to_optimize, self.config.melo.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        pexel_values = batch["image"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        labels = []
        temp = input_ids.tolist()
        for j, iter in enumerate(temp):
            for k in range(len(iter)):
                if k < batch["prompts_len"][j]:
                    iter[k] = -100
            labels.append(iter)
        labels = torch.tensor(labels)
        labels = labels.masked_fill(
            labels == 1, -100
        )


        self.losses = []
        for i in range(self.config.melo.num_iter):
            outputs = self.model.model(input_ids=input_ids, pixel_values=pexel_values, labels=labels,
                                       attention_mask=attention_mask)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            numpy_loss = loss.detach().cpu().numpy()
            self.losses.append(numpy_loss)


            LOG.info(f'Batch output loss in iter {i}: {numpy_loss:.8f},')

        with torch.no_grad():
            outputs = self.model.model(input_ids=input_ids, pixel_values=pexel_values, labels=labels,
                                       attention_mask=attention_mask)
            self.outputs[batch_index] = outputs






    def get_output(self, batch, lora_block_mapping):
        # reset batch lora_block_mapping
        if lora_block_mapping is not None:
            self.set_lora_mapping(lora_block_mapping)

        if isinstance(batch["image"], torch.Tensor):
            pixel_values = batch["image"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            labels = []
            temp = input_ids.tolist()
            for j, iter in enumerate(temp):
                for k in range(len(iter)):
                    if k < batch["prompts_len"][j]:
                        iter[k] = -100
                labels.append(iter)
            labels = torch.tensor(labels)
            labels = labels.masked_fill(
                labels == 1, -100
            )
            outputs = self.model.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels = labels)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = self.model.model(input_ids=input_ids, pixel_values=None, attention_mask=attention_mask)

        return outputs

    def generate_output(self, batch, lora_block_mapping):
        # reset batch lora_block_mapping
        self.set_lora_mapping(lora_block_mapping)
        if isinstance(batch["image"], torch.Tensor):
            pexel_values = batch["image"]
            labels = batch["labels"]
            input_ids = batch["prompt_ids"]
            outputs = self.model.generate(input_ids=input_ids, pixel_values=pexel_values)
        return outputs

    # def generate(self, *args, **kwargs):
    #     return self.model.model.generate(*args, **kwargs)




if __name__ == '__main__':
    pass


















