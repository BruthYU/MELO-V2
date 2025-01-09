import copy
import random
import importlib
import logging
from time import time
import hydra
from omegaconf import OmegaConf,open_dict
import numpy as np
import torch
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from dataset import *
from pathlib import Path
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
)
import hashlib
LOG = logging.getLogger(__name__)
class dream_trainer:
    def __init__(self, config, alg, accelerator, tokenizer, metric, data_info, subject_list, identifier_list):
        self.config = config
        self.alg = alg
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.metric = metric


        self.data_info = data_info
        self.subject_list = subject_list
        self.identifier_list = identifier_list

        self.prepare_dataset()

    def run_edit(self):
        self.alg.enable_melo()
        for train_dataset, train_dataloader in zip(self.train_dataset_list, self.train_dataloader_list):
            self.alg.edit(train_dataset, train_dataloader)
        self.alg.save_pipeline()

    def prepare_dataset(self):
        # Generate class images if prior preservation is enabled.
        for x in self.subject_list:
            assert x in self.data_info.keys(), f"No instance images of {x}"
            if self.data_info[x]["with_prior"]:
                class_images_dir = Path(self.config.base_dir, "data/class_datas", self.data_info[x]["class_name"])
                if not class_images_dir.exists():
                    class_images_dir.mkdir(parents=True)

                cur_class_images = len(list(class_images_dir.iterdir()))

                if cur_class_images < self.config.num_class_images:
                    torch_dtype = torch.float16 if self.accelerator.device.type == "cuda" else torch.float32
                    if self.config.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif self.config.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif self.config.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        self.config.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=self.config.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = self.config.num_class_images - cur_class_images
                    LOG.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(" ".join([self.config.class_prompt,self.data_info[x]["class_name"]]), num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.config.sample_batch_size)

                    sample_dataloader = self.accelerator.prepare(sample_dataloader)
                    pipeline.to(self.accelerator.device)

                    for example in tqdm(
                            sample_dataloader, desc="Generating class images",
                            disable=not self.accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        self.train_dataset_list = []
        self.train_dataloader_list = []
        for sub, id in zip(self.subject_list, self.identifier_list):
            class_images_dir = Path(self.config.base_dir, "data/class_datas", self.data_info[sub]["class_name"])
            instance_data_dir = Path(self.config.base_dir, "data/instances", sub)
            instance_prompt = " ".join([self.config.instance_prompt, id, sub.replace("_", " ")])
            class_prompt = " ".join([self.config.class_prompt, self.data_info[sub]["class_name"]])
            train_dataset = DreamBoothDataset(
                instance_data_root=instance_data_dir,
                instance_prompt=instance_prompt,
                class_data_root=class_images_dir if self.data_info[sub]["with_prior"] else None,
                class_prompt=class_prompt,
                class_num=self.config.num_class_images,
                tokenizer=self.tokenizer,
                size=self.config.resolution,
                center_crop=self.config.center_crop,
                encoder_hidden_states=None,
                class_prompt_encoder_hidden_states=None,
                tokenizer_max_length=self.config.tokenizer_max_length,
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.config.train_batch_size,
                shuffle=True,
                collate_fn=lambda examples: collate_fn(examples, self.config.with_prior_preservation),
                num_workers=self.config.dataloader_num_workers,
            )
            self.train_dataset_list.append(train_dataset)
            self.train_dataloader_list.append(train_dataloader)





