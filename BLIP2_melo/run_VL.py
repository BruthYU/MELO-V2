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
import models
from multimodal_trainer import caption_trainer, vqa_trainer

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = logging.getLogger(__name__)
@hydra.main(config_path='config', config_name='config')
def run(config):
    grace_config_keys = ['edit_lr','init_radius','expand_mode','key_id','num_edit_per_block','num_block','num_rank_per_block']
    model_config_keys = ['target_modules','grace_layer']

    GRACE_CONFIG = dict(config.grace)
    MODEL_CONFIG = dict(config.model)


    for k in grace_config_keys:
        LOG.info(f'[-GRACE CONFIG-]  {k}: {GRACE_CONFIG[k]}')
    for k in model_config_keys:
        LOG.info(f'[-MODEL CONFIG-]  {k}: {MODEL_CONFIG[k]}')

    base_dir = hydra.utils.get_original_cwd()
    with open_dict(config):
        config.base_dir = base_dir

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.task == "vqa":
        model = models.get_hf_model(config)
    elif config.task == "caption":
        model = models.get_hf_model(config)
    else:
        print(f"{config.task} task not found")

    model.to(config.device)
    processor = models.get_processor(config)

    '''
    Load Dataset
    '''
    if config.task == "caption":
        from multimodal_dataset import CaptionDataset
        batch_size = config.grace.num_edit_per_block
        train_ds = CaptionDataset('/home/hy/Yjh/EasyEdit-main/data/caption_train_edit.json',processor, config=config)
        eval_ds = CaptionDataset('/home/hy/Yjh/EasyEdit-main/data/caption_eval_edit.json',processor, config=config)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_ds.collate_fn)
    elif config.task == "vqa":
        from multimodal_dataset import VQADataset
        batch_size = config.grace.num_edit_per_block
        train_ds = VQADataset('/home/hy/Yjh/EasyEdit-main/data/vqa_train.json',processor, config=config)
        eval_ds = VQADataset('/home/hy/Yjh/EasyEdit-main/data/vqa_eval.json',processor, config=config)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_ds.collate_fn)

    
    # # test_original
    # # pip install accelerate
    # import requests
    # from PIL import Image
    # from transformers import Blip2Processor, Blip2ForConditionalGeneration

    # processor_test = Blip2Processor.from_pretrained("/home/hy/Yjh/Blip2_test/Blip2/")
    # model_test = Blip2ForConditionalGeneration.from_pretrained("/home/hy/Yjh/Blip2_test/Blip2/", device_map="auto")

    # for i, batch in tqdm(enumerate(train_loader)):
    #     pexel_values=batch["loc_image"]["image"]
    #     labels=batch["loc_image"]["labels"]
    #     input_ids=batch["loc_image"]["prompt_ids"]
    #     outputs = model_test.generate(input_ids=input_ids, pixel_values=pexel_values)
    #     re=processor_test.batch_decode(outputs, skip_special_tokens=True)
    #     ori=processor_test.batch_decode(labels, skip_special_tokens=True)
    #     print(outputs)
    #     print(re)
    #     print(ori)
    #     print(processor_test.batch_decode(input_ids, skip_special_tokens=True))
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")





    alg_module = importlib.import_module(f'algs.{config.alg}')
    AlgClass = getattr(alg_module,config.alg.upper())
    alg = AlgClass(model,config,processor)
    alg.to(config.device)
    

    # Trainer
    if config.task == "caption":
        trainer = caption_trainer(config,alg,processor,train_loader,eval_loader)
    elif config.task == "vqa":
        trainer = vqa_trainer(config,alg,processor,train_loader,eval_loader)
    

    # trainer.pre_editing_analyse()
    torch.cuda.empty_cache()
    trainer.run_edit()


if __name__ == '__main__':
    run()