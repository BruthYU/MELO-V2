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
    def __init__(self, model, config, model_processor, scale=None):
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

        '''Load Tokenizer
        '''
        self.model_processor = model_processor



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

    def set_grace_store_mode(self, mode):
        self.model.base_model.set_grace_store_mode_clip(mode=mode)


    def init_image_encoder(self):
        if self.config.image_encoder_name == 'dino':
            processor = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
            encoder = encoder.to(self.config.device)

        elif self.config.image_encoder_name == 'clip':
            from transformers import CLIPProcessor, CLIPModel
            encoder = CLIPModel.from_pretrained("/home/hy/Yjh/MELO-master/basemodel/vit")
            processor = CLIPProcessor.from_pretrained("/home/hy/Yjh/MELO-master/basemodel/vit")
        elif self.config.image_encoder_name == 'vit':
            from transformers import ViTFeatureExtractor, AutoModel
            processor = ViTFeatureExtractor.from_pretrained('/home/hy/Yjh/MELO-master/basemodel/VIT')
            encoder = AutoModel.from_pretrained('/home/hy/Yjh/MELO-master/basemodel/VIT')
        else:
            raise ValueError("Unknown Encoder Name !")

        self.image_encoder = encoder
        self.image_processor = processor

    def init_flagembedding(self, config):
        from FlagEmbedding import FlagModel
        encoder = FlagModel('/home/hy/Yjh/MELO-master/basemodel/flag-embedding/',
                          query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        self.text_encoder = encoder


    def edit(self, tokens):
        # set MELO lora_block_mapping


        # MELO_V2 could automatically identify lora parameters to be optimized
        params_to_optimize = (itertools.chain(self.model.parameters()))
        optimizer = torch.optim.Adam(params_to_optimize, self.config.melo.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


        self.losses = []
        for i in range(self.config.melo.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            setattr(self.model.get_submodule(self.grace_layer), "batch_iter", i)

            # --- pass tokens through model (including through the GRACE layer) ---
            # outputs = self.model.model(**tokens)
            # input_ids=token["iamge"]
            pexel_values = tokens["image"]
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            labels = []
            temp = input_ids.tolist()
            for j, iter in enumerate(temp):
                for k in range(len(iter)):
                    if k < tokens["prompts_len"][j]:
                        iter[k] = -100
                labels.append(iter)
            labels = torch.tensor(labels)
            labels = labels.masked_fill(
                labels == 1, -100
            )

            outputs = self.model.model(input_ids=input_ids, pixel_values=pexel_values, labels=labels,
                                       attention_mask=attention_mask)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')
        self.loss = loss  # Log final loss

        setattr(self.model.get_submodule(self.grace_layer), "training", False)

    def get_output(self, batch):
        setattr(self.model.get_submodule(self.grace_layer), "training", False)
        if isinstance(batch["image"], torch.Tensor):
            pixel_values = batch["image"]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = self.model.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = self.model.model(input_ids=input_ids, pixel_values=None, attention_mask=attention_mask)

        return outputs

    def generate_output(self, batch):
        setattr(self.model.get_submodule(self.grace_layer), "training", False)
        if isinstance(batch["image"], torch.Tensor):
            pexel_values = batch["image"]
            labels = batch["labels"]
            input_ids = batch["prompt_ids"]
            outputs = self.model.generate(input_ids=input_ids, pixel_values=pexel_values)
        return outputs

    # def generate(self, *args, **kwargs):
    #     return self.model.model.generate(*args, **kwargs)

    def get_VecDB_info(self):
        VecDB_logdict = {}
        VecDB_logdict["num_cluster"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB"))
        VecDB_logdict["conflict_num"] = getattr(self.model.get_submodule(self.grace_layer), "VecDB").conflict_num
        VecDB_logdict["forget_keys"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB").forget_keys)
        return VecDB_logdict


if __name__ == '__main__':
    pass


















