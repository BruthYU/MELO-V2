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
from metrics import compute_multimodal_edit_results

LOG = logging.getLogger(__name__)

class vqa_trainer:
    def __init__(self, config, alg, processor, train_loader, eval_loader):
        self.config = config
        self.alg = alg
        self.processor = processor
        self.train_loader = train_loader
        self.eval_loader  = eval_loader
        self.batch_size = config.grace.num_edit_per_block

    def run_edit(self):
        self.alg.enable_melo()
        self.alg.init_dino(self.config)
        self.alg.init_flagembedding(self.config)
        n_edits = 0
        batch_history = []
        loc_history = []
        total_edit_time = 0
        all_edit_time = {}
        all_HIS = {}
        all_HOLDOUT = {}
        all_UP = {}
        all_VecDB = {}
        vo=[]

        for i, batch in tqdm(enumerate(self.eval_loader)):
            if i==96:
                print("wait")
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            if n_edits < self.config.max_n_edits:
                n_edits += self.batch_size
                batch_history.append(batch)

                #To 
                with torch.no_grad():
                    self.alg.disable_melo()

                    base_outputs = self.alg.get_output(batch["loc"])
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:  
                        base_logits = base_outputs

                    base_image_outputs = self.alg.get_output(batch["loc_image"])
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    loc_dic={}
                    loc_dic["loc"]=base_logits
                    loc_dic["loc_image"]=base_image_logits
                    loc_history.append(loc_dic)

                # --- perform edit ---
                self.alg.enable_melo()
                edit_start = time()
                self.alg.get_image(batch["edit_inner"])
                self.alg.get_image_id(batch["edit_inner"])
                self.alg.edit(batch["edit_inner"])
                edit_time = time() - edit_start
                total_edit_time += edit_time

            # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    # Editing loss
                    result=compute_multimodal_edit_results(self.alg,batch,self.processor)
                    print(result)

                    self.alg.get_image(batch["loc"])
                    post_base_outputs = self.alg.get_output(batch["loc"])
                    if not isinstance(post_base_outputs, torch.Tensor):
                        post_base_logits = post_base_outputs.logits
                    else:
                        post_base_logits = post_base_outputs
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

                    self.alg.get_image(batch["loc_image"])
                    post_image_base_outputs = self.alg.get_output(batch["loc_image"])
                    if not isinstance(post_image_base_outputs, torch.Tensor):
                        post_image_base_logits = post_image_base_outputs.logits
                    else:
                        post_image_base_logits = post_image_base_outputs
                    post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                    base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices

                    info_dict = {}
                    info_dict['edit/acc'] = result["rewrite_acc"].item()
                    info_dict['inner/rephrase_acc'] = result["rephrase_acc"].item()
                    info_dict['image_rephrase/acc'] = result["image_rephrase_acc"].item()
                    info_dict['time/edit'] = edit_time
                    info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                    info_dict['image_loc/acc'] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
                    
                    LOG.info(f"Batch {i} after Editing: edit/acc: {info_dict['edit/acc']} || inner/rephrase_acc: {info_dict['inner/rephrase_acc']} || image_rephrase/acc: {info_dict['image_rephrase/acc']} || image_loc/acc: {info_dict['image_loc/acc']} || loc/acc: {info_dict['loc/acc']} || time: {edit_time} || total_time: {total_edit_time}")

                    if (i > 0 and n_edits % self.config.grace.metric_period == 0) or (i == len(self.eval_loader) - 1):
                        LOG.info(f'-------------------------    Eval all {n_edits} history edits----------------------------------')
                        averager = RunningStatAverager("val")

                        for k,iter in enumerate(batch_history):
                            result = compute_multimodal_edit_results(self.alg, iter , self.processor)
                            # Text locality
                            self.alg.get_image(iter["loc"])
                            post_base_outputs = self.alg.get_output(iter["loc"])
                            if not isinstance(post_base_outputs, torch.Tensor):
                                post_base_logits = post_base_outputs.logits
                            else:
                                post_base_logits = post_base_outputs
                            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(loc_history[k]["loc"], dim=-1), k=1, dim=-1).indices

                            # Image locality
                            self.alg.get_image(iter["loc_image"])                            
                            post_image_base_outputs = self.alg.get_output(iter["loc_image"])
                            if not isinstance(post_image_base_outputs, torch.Tensor):
                                post_image_base_logits = post_image_base_outputs.logits
                            else:
                                post_image_base_logits = post_image_base_outputs
                            post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                            base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(loc_history[k]['loc_image'], dim=-1), k=10, dim=-1).indices



                            result["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
                            result["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]

                            averager.add(result)
                            
                        stats = averager.average()

                        LOG.info(f'{stats}')



        # with open(f'log.pkl', 'wb') as f:
        #     pickle.dump(
        #         {'all_UP': all_UP, 'all_HIS': all_HIS, 'all_HOLDOUT': all_HOLDOUT, 'all_edit_time': all_edit_time,
        #          'all_VecDB': all_VecDB}, f)

        # LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")
                        



