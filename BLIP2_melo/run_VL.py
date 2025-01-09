import copy
import random
import importlib
import logging
from time import time
import hydra
from omegaconf import OmegaConf, open_dict
import numpy as np
import torch
from utils import *

from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import models
from multimodal_trainer import vqa_trainer, caption_trainer

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='config')
def run(config):
    melo_config_keys = ['edit_lr', 'init_radius', 'expand_mode', 'key_id', 'num_edit_per_block', 'num_block',
                        'num_rank_per_block']
    model_config_keys = ['target_modules']

    MELO_CONFIG = dict(config.melo)
    MODEL_CONFIG = dict(config.model)

    for k in melo_config_keys:
        LOG.info(f'[-GRACE CONFIG-]  {k}: {MELO_CONFIG[k]}')
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
        from metrics import compute_multimodal_edit_results
        metric = compute_multimodal_edit_results
        batch_size = config.melo.num_edit_per_block
        eval_ds = CaptionDataset('/home/hy/Yjh/MELO-master/FlagEmbedding_CLIP/caption_eval_edit.json', processor,
                                 config=config)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_ds.collate_fn)
    elif config.task == "vqa":
        from multimodal_dataset import VQADataset
        from metrics import compute_multimodal_edit_results
        metric = compute_multimodal_edit_results
        batch_size = config.melo.num_edit_per_block
        eval_ds = VQADataset('/home/hy/Yjh/EasyEdit-main/data/vqa_eval.json',
                             processor, config=config, split="eval")
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_ds.collate_fn)

    alg_module = importlib.import_module(f'algs.{config.alg}')
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, processor)
    alg.to(config.device)

    # Trainer
    if config.task == "caption":
        trainer = caption_trainer(config, alg, metric, None, eval_loader)
    elif config.task == "vqa":
        trainer = vqa_trainer(config, alg, metric, None, eval_loader)

    # trainer.pre_editing_analyse()
    torch.cuda.empty_cache()
    trainer.run_edit()


if __name__ == '__main__':
    run()