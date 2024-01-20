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
from database.router import Router
from database.tools import NO_LORA

LOG = logging.getLogger(__name__)


class vqa_trainer:
    def __init__(self, config, alg, metric, train_loader, eval_loader):
        self.config = config
        self.alg = alg
        self.metric = metric
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.batch_size = config.melo.num_edit_per_block
        self.router = Router(self.config)

    def run_edit(self):
        self.alg.enable_melo()
        n_edits = 0
        batch_history = []
        loc_history = []
        total_edit_time = 0
        log_dict = {}

        for i, batch in tqdm(enumerate(self.eval_loader)):
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            if n_edits < self.config.max_n_edits:
                n_edits += self.batch_size
                batch_history.append(batch)

                '''
                Record Local Output (disable melo)
                '''
                self.alg.disable_melo()
                with torch.no_grad():
                    base_outputs = self.alg.get_output(batch["loc"], None)
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:
                        base_logits = base_outputs

                    base_image_outputs = self.alg.get_output(batch["loc_image"], None)
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    loc_dic = {}
                    loc_dic["loc"] = base_logits
                    loc_dic["loc_image"] = base_image_logits
                    loc_history.append(loc_dic)

                '''
                Perform Edit (enable melo)
                '''
                self.alg.enable_melo()
                edit_start = time()
                LOG.info(f"------[Vector Database Operation for Batch {i}]---------")
                batch_query, batch_query_vision = self.router.batch_embed(batch["edit_inner"])
                self.router.database_batch_add(batch_query, batch_query_vision)
                self.alg.set_lora_mapping([i] * len(batch["edit_inner"]["labels"]))
                self.alg.edit(batch["edit_inner"], i)
                edit_time = time() - edit_start
                total_edit_time += edit_time

                with torch.no_grad():
                    if (i >= 0 and n_edits % self.config.melo.metric_period == 0) or (i == len(self.eval_loader) - 1):
                        LOG.info(
                            f'-------------------------    Eval all {n_edits} history edits----------------------------------')
                        averager = RunningStatAverager("val")

                        for k, eval_batch in enumerate(batch_history):
                            # LOG.info(f"---[Alg last outputs]----")
                            # LOG.info(self.alg.outputs[k].logits.shape)
                            # print(self.alg.outputs[k].logits[:, :, 15])

                            result = self.metric(self.alg, self.router, eval_batch)
                            # Text locality
                            lora_block_mapping = self.router.get_lora_mapping(eval_batch["loc"])
                            post_base_outputs = self.alg.get_output(eval_batch["loc"], lora_block_mapping)

                            if not isinstance(post_base_outputs, torch.Tensor):
                                post_base_logits = post_base_outputs.logits
                            else:
                                post_base_logits = post_base_outputs
                            post_base_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                            base_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(loc_history[k]["loc"], dim=-1), k=1, dim=-1).indices

                            # Image locality
                            lora_block_mapping = self.router.get_lora_mapping(eval_batch["loc_image"])
                            post_image_base_outputs = self.alg.get_output(eval_batch["loc_image"], lora_block_mapping)
                            if not isinstance(post_image_base_outputs, torch.Tensor):
                                post_image_base_logits = post_image_base_outputs.logits
                            else:
                                post_image_base_logits = post_image_base_outputs
                            post_image_base_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                            base_image_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(loc_history[k]['loc_image'], dim=-1), k=10, dim=-1).indices
                            result["loc/acc"] = sum(
                                post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1)) / \
                                                post_base_logits_softmax_top_k.view(-1).shape[0]
                            result["image_loc/acc"] = sum(
                                post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(
                                    -1)) / post_image_base_logits_softmax_top_k.view(-1).shape[0]

                            averager.add(result)

                        stats = averager.average()
                        LOG.info(f'{stats}')

                        'step log_dict'
                        for k, v in stats.items():
                            if k not in log_dict.keys():
                                log_dict[k] = {}
                            log_dict[k][n_edits] = v

        with open(f'log.pkl', 'wb') as f:
            pickle.dump(log_dict, f)
        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")


class caption_trainer:
    def __init__(self, config, alg, metric, train_loader, eval_loader):
        self.config = config
        self.alg = alg
        self.metric = metric
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.batch_size = config.melo.num_edit_per_block
        self.router = Router(self.config)

    def run_edit(self):
        self.alg.enable_melo()
        n_edits = 0
        batch_history = []
        loc_history = []
        total_edit_time = 0
        log_dict = {}

        for i, batch in tqdm(enumerate(self.eval_loader)):
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            if n_edits < self.config.max_n_edits:
                n_edits += self.batch_size
                batch_history.append(batch)

                '''
                Record Local Output (disable melo)
                '''
                self.alg.disable_melo()
                with torch.no_grad():
                    base_outputs = self.alg.get_output(batch["loc"], None)
                    if not isinstance(base_outputs, torch.Tensor):
                        base_logits = base_outputs.logits
                    else:
                        base_logits = base_outputs

                    base_image_outputs = self.alg.get_output(batch["loc_image"], None)
                    if not isinstance(base_image_outputs, torch.Tensor):
                        base_image_logits = base_image_outputs.logits
                    else:
                        base_image_logits = base_image_outputs
                    loc_dic = {}
                    loc_dic["loc"] = base_logits
                    loc_dic["loc_image"] = base_image_logits
                    loc_history.append(loc_dic)

                '''
                Perform Edit (enable melo)
                '''
                self.alg.enable_melo()
                edit_start = time()
                LOG.info(f"------[Vector Database Operation for Batch {i}]---------")
                batch_query, batch_query_vision = self.router.batch_embed(batch["edit_inner"])
                self.router.database_batch_add(batch_query, batch_query_vision)
                self.alg.set_lora_mapping([i] * len(batch["edit_inner"]["labels"]))
                self.alg.edit(batch["edit_inner"], i)
                edit_time = time() - edit_start
                total_edit_time += edit_time

                with torch.no_grad():
                    if (i >= 0 and n_edits % self.config.melo.metric_period == 0) or (i == len(self.eval_loader) - 1):
                        LOG.info(
                            f'-------------------------    Eval all {n_edits} history edits----------------------------------')
                        averager = RunningStatAverager("val")

                        for k, eval_batch in enumerate(batch_history):
                            # LOG.info(f"---[Alg last outputs]----")
                            # LOG.info(self.alg.outputs[k].logits.shape)
                            # print(self.alg.outputs[k].logits[:, :, 15])

                            result = self.metric(self.alg, self.router, eval_batch)
                            # Text locality
                            lora_block_mapping = self.router.get_lora_mapping(eval_batch["loc"])
                            post_base_outputs = self.alg.get_output(eval_batch["loc"], lora_block_mapping)

                            if not isinstance(post_base_outputs, torch.Tensor):
                                post_base_logits = post_base_outputs.logits
                            else:
                                post_base_logits = post_base_outputs
                            post_base_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                            base_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(loc_history[k]["loc"], dim=-1), k=1, dim=-1).indices

                            # Image locality
                            lora_block_mapping = self.router.get_lora_mapping(eval_batch["loc_image"])
                            post_image_base_outputs = self.alg.get_output(eval_batch["loc_image"], lora_block_mapping)
                            if not isinstance(post_image_base_outputs, torch.Tensor):
                                post_image_base_logits = post_image_base_outputs.logits
                            else:
                                post_image_base_logits = post_image_base_outputs
                            post_image_base_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
                            base_image_logits_softmax_top_k = torch.topk(
                                torch.nn.functional.softmax(loc_history[k]['loc_image'], dim=-1), k=10, dim=-1).indices
                            result["loc/acc"] = sum(
                                post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1)) / \
                                                post_base_logits_softmax_top_k.view(-1).shape[0]
                            result["image_loc/acc"] = sum(
                                post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(
                                    -1)) / post_image_base_logits_softmax_top_k.view(-1).shape[0]

                            averager.add(result)

                        stats = averager.average()
                        LOG.info(f'{stats}')

                        'step log_dict'
                        for k, v in stats.items():
                            if k not in log_dict.keys():
                                log_dict[k] = {}
                            log_dict[k][n_edits] = v

        with open(f'log.pkl', 'wb') as f:
            pickle.dump(log_dict, f)
        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")