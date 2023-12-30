import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate


class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, vis_root=None, rephrase_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.rephrase_root = rephrase_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        # self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor):
        self.vis_processor = vis_processor
        # self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
    



##Coco_caption
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from processor.base_dataset import BaseDataset
from processor.blip_processors import BlipImageEvalProcessor
from utils import dict_to
from PIL import Image
import random
import typing
import torch
import transformers

# class CaptionDataset(BaseDataset):
#     def __init__(self, data_dir: str, processor, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """
#         # get tokenizer and vis_processor
#         vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
#         if (config is not None and hasattr(config, 'tokenizer_name')):
#             tok_name = (
#                 config.tokenizer_name
#                 if config.tokenizer_name is not None
#                 else config.name
#             )
#             tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
#                 tok_name, trust_remote_code=True
#             )            
#             if tokenizer.pad_token == None or tokenizer.pad_token == '':
#                 tokenizer.pad_token = tokenizer.eos_token  
                
#         vis_root = config.coco_image
#         rephrase_root = config.rephrase_image
#         super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

#         self.config = config
#         self.tok = tokenizer
#         self.max_length = 32
#         self.processor = processor

#         self.prompt = "Question: {} Short answer: "

#         data = []
#         for i, record in enumerate(self.annotation):
            
#             if record['alt'] == "":
#                 continue
            
#             #image_path = os.path.join(self.vis_root, record["image"])
#             # rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
#             # locality_image_path = os.path.join(self.vis_root, record['m_loc'])

#             image_path = "/home/hy/Yjh/EasyEdit-main/"+record["image"]
#             rephrase_image_path = "/home/hy/Yjh/EasyEdit-main/"+record["image_rephrase"]
#             locality_image_path = "/home/hy/Yjh/EasyEdit-main/"+ record['m_loc']
            
#             image = Image.open(image_path).convert("RGB")
#             rephrase_image = Image.open(rephrase_image_path).convert("RGB")
#             locality_image = Image.open(locality_image_path).convert("RGB")

#             # image = self.vis_processor(image)
#             # rephrase_image = self.vis_processor(rephrase_image)  
#             # locality_image = self.vis_processor(locality_image)   
                      
#             item = {
#                 'prompt': record['src'],
#                 'pred': record['pred'],
#                 'target': record['alt'],
#                 'rephrase_prompt': record['rephrase'],
#                 'image': image,
#                 'image_rephrase': rephrase_image,
#                 'cond': "{} >> {} || {}".format(
#                     record['pred'],
#                     record['alt'],
#                     record['src']
#                 )
#             }
            
#             item['locality_prompt'] = record['loc']
#             item['locality_ground_truth'] = record['loc_ans']
            
#             item['multimodal_locality_image'] = locality_image
#             item['multimodal_locality_prompt'] = record['m_loc_q']
#             item['multimodal_locality_ground_truth'] = record['m_loc_a']
#             data.append(item)
            
#         if size is not None:
#             data = data[:size]        
#         self._data = data

#     def __getitem__(self, index):
#         return self._data[index]
    
#     def __len__(self):
#         return len(self._data)

#     def collate_fn(self, batch):
#         src = [b['prompt'] for b in batch]
#         trg = [b['target'] for b in batch]
#         cond = [b['cond'] for b in batch]
#         rephrase = [b['rephrase_prompt'] for b in batch]
#         image = [b['image'] for b in batch]
#         image_rephrase = [b['image_rephrase'] for b in batch]
#         loc_q = [b["locality_prompt"] for b in batch]
#         loc_a = [b["locality_ground_truth"] for b in batch]
#         m_loc_image = [b['multimodal_locality_image'] for b in batch]
#         m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
#         m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

#         image=[self.processor(images=image, padding=True, return_tensors="pt")['pixel_values'].squeeze()]
#         image_rephrase=[self.processor(images=image_rephrase, padding=True, return_tensors="pt")['pixel_values'].squeeze()]
#         m_loc_image=[self.processor(images=m_loc_image, padding=True, return_tensors="pt")['pixel_values'].squeeze()]


        
#         # edit_inner
#         edit_inner = {}
#         #edit_inner['image'] = torch.stack(image, dim=0)
#         edit_inner['image'] = torch.stack(image, dim=0).squeeze()

#         edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
#         edit_inner['labels'] = trg
#         if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
#             edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
#             #edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
#             edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#             edit_inner['input_ids'] = self.tok(edit_inner['text_input'], add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#         else:
#             edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
#             #edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
#             edit_inner['labels'] = self.tok(trg, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
#         # edit_outer
#         edit_outer = {}
#         #edit_outer['image'] = torch.stack(image, dim=0)
#         edit_outer['image'] = torch.stack(image, dim=0).squeeze()

#         edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
#         edit_outer['labels'] = trg
#         if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
#             edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
#             #edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
#             edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#             edit_outer['input_ids'] = self.tok(edit_outer['text_input'], add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#         else:
#             edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
#             #edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
#             edit_outer['labels'] = self.tok(trg, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            
#         # edit_outer_image
#         edit_outer_image = {}
#         #edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
#         edit_outer_image['image'] = torch.stack(image_rephrase, dim=0).squeeze()

#         edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
#         edit_outer_image['labels'] = trg
#         if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
#             edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
#             #edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
#             edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#             edit_outer_image['input_ids'] = self.tok(edit_outer_image['text_input'], add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#         else:
#             edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
#             #edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
#             edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
#         # loc
#         loc = {}
#         loc['image'] = None
#         loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
#         loc['labels'] = loc_a
#         if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
#             loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
#             #loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
#             loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#             loc['input_ids'] = self.tok(loc['text_input'], add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#         else:
#             loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
#             #loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
#             loc['labels'] = self.tok(loc_a, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
#         # m_loc
#         loc_image = {}
#         #loc_image['image'] = torch.stack(m_loc_image, dim=0)
#         loc_image['image'] = torch.stack(m_loc_image, dim=0).squeeze()
        
#         loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
#         loc_image['labels'] = m_loc_a
#         if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
#             loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
#             #loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
#             loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#             loc_image['input_ids'] = self.tok(loc_image['text_input'], add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
#         else:
#             loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
#             #loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]
#             loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",padding=True, truncation=True)["input_ids"]

#         # cond
#         cond = self.tok(
#             cond,
#             return_tensors="pt",
#             padding=True,
#             max_length=self.max_length,
#             truncation=True,
#         ).to(self.config.device)
        
#         batch = {
#             "edit_inner": edit_inner,
#             "edit_outer": edit_outer,
#             "edit_outer_image": edit_outer_image,
#             "loc": loc,
#             "loc_image": loc_image,
#             "cond": cond
#         }
#         return dict_to(batch, self.config.device)





class CaptionDataset(BaseDataset):
    def __init__(self, data_dir: str, processor, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32
        self.processor = processor

        self.prompt = "Question: {} Short answer: "

        data = []
        for i, record in enumerate(self.annotation):
            
            if record['alt'] == "":
                continue
            
            #image_path = os.path.join(self.vis_root, record["image"])
            # rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            # locality_image_path = os.path.join(self.vis_root, record['m_loc'])

            image_path = "/home/hy/Yjh/EasyEdit-main/"+record["image"]
            rephrase_image_path = "/home/hy/Yjh/EasyEdit-main/"+record["image_rephrase"]
            locality_image_path = "/home/hy/Yjh/EasyEdit-main/"+ record['m_loc']
            
            image = Image.open(image_path).convert("RGB")
            rephrase_image = Image.open(rephrase_image_path).convert("RGB")
            locality_image = Image.open(locality_image_path).convert("RGB")

            # image = self.vis_processor(image)
            # rephrase_image = self.vis_processor(rephrase_image)  
            # locality_image = self.vis_processor(locality_image)   
                      
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_rephrase': rephrase_image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            data.append(item)
            
        if size is not None:
            data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]



        image=[self.processor(images=image, padding=True, return_tensors="pt")['pixel_values'].squeeze()]
        image_rephrase=[self.processor(images=image_rephrase, padding=True, return_tensors="pt")['pixel_values'].squeeze()]
        m_loc_image=[self.processor(images=m_loc_image, padding=True, return_tensors="pt")['pixel_values'].squeeze()]



        
        # edit_inner
        edit_inner = {}
        #edit_inner['image'] = torch.stack(image, dim=0)
        if len(src)==1:
            edit_inner['image'] = torch.stack(image, dim=0)
        else:
            edit_inner['image'] = torch.stack(image, dim=0).squeeze()


        #edit_inner['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['text_input'] = ["".join([s, t]) for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_inner['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            #edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_inner['input_ids'] = self.tok(edit_inner['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_inner['prompt_ids'] = self.tok(src, add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            #edit_inner['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            edit_inner['labels'] = self.tok(trg, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
        # edit_outer
        edit_outer = {}
        #edit_outer['image'] = torch.stack(image, dim=0)
        if len(src)==1:
            edit_outer['image'] = torch.stack(image, dim=0)
        else:
            edit_outer['image'] = torch.stack(image, dim=0).squeeze()


        #edit_outer['text_input'] = [" ".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['text_input'] = ["".join([r, t]) for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer['prompts_len'] = [len(self.tok.encode(r, add_special_tokens=False)) for r in rephrase]
            #edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer['input_ids'] = self.tok(edit_outer['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer['prompt_ids'] = self.tok(rephrase, add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(r)) for r in rephrase]
            #edit_outer['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            edit_outer['labels'] = self.tok(trg, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        #edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        

        if len(src)==1:
            edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        else:
            edit_outer_image['image'] = torch.stack(image_rephrase, dim=0).squeeze()

        #edit_outer_image['text_input'] = [" ".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['text_input'] = ["".join([s, t]) for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s, add_special_tokens=False)) for s in src]
            #edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer_image['input_ids'] = self.tok(edit_outer_image['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer_image['prompt_ids'] = self.tok(src, add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(s)) for s in src]
            #edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",)["input_ids"]
            edit_outer_image['labels'] = self.tok(trg, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
        # loc
        loc = {}
        loc['image'] = None
        #loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['text_input'] = ["".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            #loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            loc['input_ids'] = self.tok(loc['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            #loc['labels'] = self.tok(loc_a, return_tensors="pt",)["input_ids"]
            loc['labels'] = self.tok(loc_a, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
        # m_loc
        loc_image = {}
        #
        

        if len(src)==1:
            loc_image['image'] = torch.stack(m_loc_image, dim=0)
        else:
            loc_image['image'] = torch.stack(m_loc_image, dim=0).squeeze()

        
        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        if self.config.model_name == "minigpt4" or self.config.model_name == "blip2":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            #loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",)["input_ids"]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            loc_image['input_ids'] = self.tok(loc_image['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            temp = [self.prompt.format(m_q) for m_q in m_loc_q]
            loc_image['prompt_ids'] = self.tok(temp, add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            #loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",)["input_ids"]
            loc_image['labels'] = self.tok(m_loc_a, return_tensors="pt",padding=True, truncation=True)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
    


class VQADataset(BaseDataset):
    def __init__(self, data_dir: str, processor, size:  typing.Optional[int] = None, config=None, *args, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # get tokenizer and vis_processor
        vis_processor = BlipImageEvalProcessor(image_size=364, mean=None, std=None)
        if (config is not None and hasattr(config, 'tokenizer_name')):
            tok_name = (
                config.tokenizer_name
                if config.tokenizer_name is not None
                else config.name
            )
            tokenizer = getattr(transformers, config.tokenizer_class).from_pretrained(
                tok_name, trust_remote_code=True
            )            
            if tokenizer.pad_token == None or tokenizer.pad_token == '':
                tokenizer.pad_token = tokenizer.eos_token  
                
        vis_root = config.coco_image
        rephrase_root = config.rephrase_image
        super().__init__(vis_processor, vis_root, rephrase_root, [data_dir])

        self.config = config
        self.tok = tokenizer
        self.max_length = 32
        self.processor=processor

        self.prompt = "Question: {} Short answer:"

        data = []
        for i, record in enumerate(self.annotation):
            
            if record['alt'] == "":
                continue
            
            # image_path = os.path.join(self.vis_root, record["image"])
            # rephrase_image_path = os.path.join(self.rephrase_root, record["image_rephrase"])
            # locality_image_path = os.path.join(self.vis_root, record['m_loc'])

            image_path = "/home/hy/Yjh/EasyEdit-main/"+record["image"]
            rephrase_image_path = "/home/hy/Yjh/EasyEdit-main/"+record["image_rephrase"]
            locality_image_path = "/home/hy/Yjh/EasyEdit-main/"+ record['m_loc']
            
            image = Image.open(image_path).convert("RGB")
            rephrase_image = Image.open(rephrase_image_path).convert("RGB")
            locality_image = Image.open(locality_image_path).convert("RGB")

            # image = self.vis_processor(image)
            # rephrase_image = self.vis_processor(rephrase_image)  
            # locality_image = self.vis_processor(locality_image)  
                      
            item = {
                'prompt': record['src'],
                'pred': record['pred'],
                'target': record['alt'],
                'rephrase_prompt': record['rephrase'],
                'image': image,
                'image_rephrase': rephrase_image,
                'cond': "{} >> {} || {}".format(
                    record['pred'],
                    record['alt'],
                    record['src']
                )
            }
            
            item['locality_prompt'] = record['loc']
            item['locality_ground_truth'] = record['loc_ans']
            
            item['multimodal_locality_image'] = locality_image
            item['multimodal_locality_prompt'] = record['m_loc_q']
            item['multimodal_locality_ground_truth'] = record['m_loc_a']
            data.append(item)
            
        if size is not None:
            data = data[:size]        
        self._data = data

    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)

    def collate_fn(self, batch):
        src = [b['prompt'] for b in batch]
        trg = [b['target'] for b in batch]
        cond = [b['cond'] for b in batch]
        rephrase = [b['rephrase_prompt'] for b in batch]
        image = [b['image'] for b in batch]
        image_rephrase = [b['image_rephrase'] for b in batch]
        loc_q = [b["locality_prompt"] for b in batch]
        loc_a = [b["locality_ground_truth"] for b in batch]
        m_loc_image = [b['multimodal_locality_image'] for b in batch]
        m_loc_q = [b['multimodal_locality_prompt'] for b in batch]
        m_loc_a = [b['multimodal_locality_ground_truth'] for b in batch]

        image=[self.processor(images=image, padding=True, return_tensors="pt")['pixel_values'].squeeze()]
        image_rephrase=[self.processor(images=image_rephrase, padding=True, return_tensors="pt")['pixel_values'].squeeze()]
        m_loc_image=[self.processor(images=m_loc_image, padding=True, return_tensors="pt")['pixel_values'].squeeze()]


        
        # edit_inner
        edit_inner = {}
        #edit_inner['image'] = torch.stack(image, dim=0)
        if len(src)==1:
            edit_inner['image'] = torch.stack(image, dim=0)
        else:
            edit_inner['image'] = torch.stack(image, dim=0).squeeze()


        edit_inner['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        edit_inner['labels'] = trg
        if self.config.model_name == "minigpt4":
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_inner['input_ids'] = self.tok(edit_inner['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            edit_inner['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_inner['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_inner['input_ids'] = self.tok(edit_inner['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
        # edit_outer
        edit_outer = {}
        if len(src)==1:
            edit_outer['image'] = torch.stack(image, dim=0)
        else:
            edit_outer['image'] = torch.stack(image, dim=0).squeeze()
        edit_outer['text_input'] = [self.prompt.format(r) + f"{t}" for r, t in zip(rephrase, trg)]
        edit_outer['labels'] = trg
        if self.config.model_name == "minigpt4":
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r), add_special_tokens=False)) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer['input_ids'] = self.tok(edit_outer['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            edit_outer['prompts_len'] = [len(self.tok.encode(self.prompt.format(r))) for r in rephrase]
            edit_outer['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer['input_ids'] = self.tok(edit_outer['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            
        # edit_outer_image
        edit_outer_image = {}
        if len(src)==1:
            edit_outer_image['image'] = torch.stack(image_rephrase, dim=0)
        else:
            edit_outer_image['image'] = torch.stack(image_rephrase, dim=0).squeeze()
        edit_outer_image['text_input'] = [self.prompt.format(s) + f"{t}" for s, t in zip(src, trg)]
        edit_outer_image['labels'] = trg
        if self.config.model_name == "minigpt4":
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s), add_special_tokens=False)) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer_image['input_ids'] = self.tok(edit_outer_image['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            edit_outer_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(s))) for s in src]
            edit_outer_image['labels'] = self.tok(trg, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            edit_outer_image['input_ids'] = self.tok(edit_outer_image['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
        # loc
        loc = {}
        loc['image'] = None
        loc['text_input'] = [" ".join([q, a]) for q, a in zip(loc_q, loc_a)]
        loc['labels'] = loc_a
        if self.config.model_name == "minigpt4":
            loc['prompts_len'] = [len(self.tok.encode(q, add_special_tokens=False)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            loc['input_ids'] = self.tok(loc['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            loc['prompts_len'] = [len(self.tok.encode(q)) for q in loc_q]
            loc['labels'] = self.tok(loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            loc['input_ids'] = self.tok(loc['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        
        # m_loc
        loc_image = {}
        if len(src)==1:
            loc_image['image'] = torch.stack(m_loc_image, dim=0)
        else:
            loc_image['image'] = torch.stack(m_loc_image, dim=0).squeeze()

        loc_image['text_input'] = [self.prompt.format(q) + a for q, a in zip(m_loc_q, m_loc_a)]
        loc_image['labels'] = m_loc_a
        if self.config.model_name == "minigpt4":
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q), add_special_tokens=False)) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            loc_image['input_ids'] = self.tok(loc_image['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]
        else:
            loc_image['prompts_len'] = [len(self.tok.encode(self.prompt.format(q))) for q in m_loc_q]
            loc_image['labels'] = self.tok(m_loc_a, add_special_tokens=False, return_tensors="pt",padding=True, truncation=True)["input_ids"]
            loc_image['input_ids'] = self.tok(loc_image['text_input'], add_special_tokens=True, return_tensors="pt",padding=True, truncation=True)["input_ids"]

        # cond
        cond = self.tok(
            cond,
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,
        ).to(self.config.device)
        
        batch = {
            "edit_inner": edit_inner,
            "edit_outer": edit_outer,
            "edit_outer_image": edit_outer_image,
            "loc": loc,
            "loc_image": loc_image,
            "cond": cond
        }
        return dict_to(batch, self.config.device)
