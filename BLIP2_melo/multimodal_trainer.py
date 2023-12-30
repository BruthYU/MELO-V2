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

class zsre_trainer:
    def __init__(self, config, alg, tokenize, metric, edit_loader, upstream_loader, edit_holdout_loader):
        self.config = config
        self.alg = alg
        self.tokenize = tokenize
        self.metric = metric
        self.edit_loader = edit_loader
        self.upstream_loader  = upstream_loader
        self.edit_holdout_loader = edit_holdout_loader
        self.batch_size = config.grace.num_edit_per_block

    def pre_editing_analyse(self):
        self.alg.disable_melo()

        with torch.no_grad():
            metric_dict = {'F1': [], 'ACC': []}
            for batch in iter(self.edit_loader):
                edit_input = self.tokenize(batch, self.alg.model_tok, self.config['device'])
                f1, acc = self.metric(self.alg, edit_input)
                metric_dict['F1'].append(f1)
                metric_dict['ACC'].append(acc)
            original_f1 = torch.Tensor(metric_dict['F1']).nanmean()
            original_acc = torch.Tensor(metric_dict['ACC']).nanmean()
            LOG.info(
                f'Original average performance on edit set: F1: {original_f1.item():.4f} || ACC: {original_acc.item():.4f}')

            TRR_dict = {'F1': [], 'ACC': []}
            for up_batch in iter(self.upstream_loader):
                upstream_input = self.tokenize(up_batch, self.alg.model_tok, self.config['device'])
                up_f1, up_acc = self.metric(self.alg, upstream_input)
                TRR_dict['F1'].append(up_f1)
                TRR_dict['ACC'].append(up_acc)
            upstream_f1 = torch.Tensor(TRR_dict['F1']).nanmean()
            upstream_acc = torch.Tensor(TRR_dict['ACC']).nanmean()
            LOG.info(
                f'Original average performance on upstream set: F1: {upstream_f1.item():.4f} || ACC: {upstream_acc.item():.4f}')

    def run_edit(self):
        # --- editing start ---
        self.alg.enable_melo()
        n_edits = 0
        batch_history = []
        total_edit_time = 0
        all_edit_time = {}
        all_HIS = {}
        all_HOLDOUT = {}
        all_UP = {}
        all_VecDB = {}

        for i, batch in tqdm(enumerate(self.edit_loader)):
            if i == 25:
                print(i)
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            tokens = self.tokenize(batch, self.alg.model_tok, self.config['device'])
            if n_edits < self.config.max_n_edits:
                n_edits += self.batch_size
                batch_history.append(tokens)

                # --- perform edit ---
                edit_start = time()
                self.alg.edit(tokens)
                edit_time = time() - edit_start
                total_edit_time += edit_time

                # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    ES_f1, ES_acc = self.metric(self.alg, tokens)
                    LOG.info(f'Batch {i} after Editing: F1: {ES_f1} || ACC: {ES_acc}')

                    if (i > 0 and n_edits % self.config.grace.metric_period == 0) or (i == len(self.edit_loader) - 1):
                        LOG.info(
                            f'-------------------------    Eval all {n_edits} history edits----------------------------------')
                        if self.config.task == 'qa':
                            holdout = [self.metric(self.alg, self.tokenize(e, self.alg.model_tok, self.config['device'])) for e in
                                       iter(self.edit_holdout_loader)]
                            holdout_f1 = torch.tensor([x[0] for x in holdout]).nanmean()
                            holdout_acc = torch.tensor([x[1] for x in holdout]).nanmean()
                        else:
                            pass

                        HISTORY = [self.metric(self.alg, tokens) for tokens in batch_history]
                        HISTORY_f1 = torch.tensor([x[0] for x in HISTORY]).nanmean()
                        HISTORY_acc = torch.tensor([x[1] for x in HISTORY]).nanmean()

                        UP = [self.metric(self.alg, self.tokenize(e, self.alg.model_tok, self.config["device"], test=True)) for e in
                              iter(self.upstream_loader)]
                        UP_f1 = torch.tensor([x[0] for x in UP]).nanmean()
                        UP_acc = torch.tensor([x[1] for x in UP]).nanmean()
                        # --- Log metrics and push to Weights & Biases ---
                        log_dict["UP"] = {'UP_f1': UP_f1.item(), 'UP_acc': UP_acc.item()}  # Test Retention Rate
                        log_dict["HIS"] = {'HIS_f1': HISTORY_f1.item(),
                                           'HIS_acc': HISTORY_acc.item()}  # Error Retention Rate
                        log_dict["ES"] = {'ES_f1': ES_f1, 'ES_acc': ES_acc}  # Edit Success
                        log_dict["train_time"] = edit_time / 60  # Time it takes to make one edit
                        log_dict["edit"] = batch["text"]  # Raw edit input
                        log_dict["edit_label"] = batch["labels"]  # Raw edit label
                        log_dict["n_edits"] = n_edits  # Raw edit label
                        log_dict['holdout'] = {'holdout_f1': holdout_f1.item(), 'holdout_acc': holdout_acc.item()}
                        print(f"Number of edits {n_edits}")
                        for k in log_dict:
                            LOG.info(f"[+eval result+]{k}: {log_dict[k]}")

                        all_UP[n_edits] = log_dict["UP"]
                        all_HIS[n_edits] = log_dict["HIS"]
                        all_HOLDOUT[n_edits] = log_dict["holdout"]
                        all_edit_time[n_edits] = total_edit_time
                        VecDB_info = self.alg.get_VecDB_info()
                        for k in VecDB_info:
                            LOG.info(f"[+VecDB Info+]{k}: {VecDB_info[k]}")
                        all_VecDB[n_edits] = VecDB_info
                        pass

        with open(f'log.pkl', 'wb') as f:
            pickle.dump(
                {'all_UP': all_UP, 'all_HIS': all_HIS, 'all_HOLDOUT': all_HOLDOUT, 'all_edit_time': all_edit_time,
                 'all_VecDB': all_VecDB}, f)

        LOG.info(f"[**Total Edit Time**] {total_edit_time / 60} mins")



def multiclass_log_probs(pred, targ, shift=True):
    NULL_TOKEN = 0  # a placeholder used for masked target locations
    pred = pred.to(torch.float32)

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        pred = pred[:, -targ.size(1):]
        targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    
    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()

    acc = correct.sum() / num_non_padding

    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": -log_prob,
    }



def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (
                pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))
            ).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError



class caption_trainer:
    def __init__(self, config, alg, processor, train_loader, eval_loader):
        self.config = config
        self.alg = alg
        self.processor = processor
        self.train_loader = train_loader
        self.eval_loader  = eval_loader
        self.batch_size = config.grace.num_edit_per_block

    # def run_edit(self):
    #     # --- editing start ---
    #     self.alg.enable_melo()
    #     n_edits = 0
    #     batch_history = []
    #     total_edit_time = 0
    #     all_edit_time = {}
    #     all_HIS = {}
    #     all_HOLDOUT = {}
    #     all_UP = {}
    #     all_VecDB = {}

    #     for i, batch in tqdm(enumerate(self.train_loader)):
    #         LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
    #         if n_edits < self.config.max_n_edits:
    #             n_edits += self.batch_size
    #             batch_history.append(batch)


    #             with torch.no_grad():
    #                 ## 测评文本端的局部性
    #                 # base_outputs = self.alg.model(batch["loc"])
    #                 # if not isinstance(base_outputs, torch.Tensor):
    #                 #     base_logits = base_outputs.logits
    #                 # else:  
    #                 #     base_logits = base_outputs
                        
    #                 base_image_outputs = self.alg.get_output(batch["loc_image"])
    #                 if not isinstance(base_image_outputs, torch.Tensor):
    #                     base_image_logits = base_image_outputs.logits
    #                 else:
    #                     base_image_logits = base_image_outputs

    #             # --- perform edit ---
    #             edit_start = time()
    #             self.alg.edit(batch["edit_inner"])
    #             edit_time = time() - edit_start
    #             total_edit_time += edit_time

    #         # --- Compute and log metrics ---
    #             log_dict = {}
    #             with torch.no_grad():
    #                 # Editing loss
    #                 post_edit_outputs=self.alg.get_output(batch["edit_outer"])
    #                 post_batch_labels = batch["edit_outer"]["labels"]
    #                 if not isinstance(post_edit_outputs, torch.Tensor):
    #                     post_edit_logits = post_edit_outputs.logits
    #                 else:
    #                     post_edit_logits = post_edit_outputs

    #                 # rephrase image
    #                 post_image_edit_outputs = self.alg.get_output(batch["edit_outer_image"])
    #                 post_image_batch_labels = batch["edit_outer_image"]["labels"]
    #                 if not isinstance(post_image_edit_outputs, torch.Tensor):
    #                     post_image_edit_logits = post_image_edit_outputs.logits
    #                 else:
    #                     post_image_edit_logits = post_image_edit_outputs
                        
    #                 inner_edit_outputs = self.alg.get_output(batch["edit_inner"])
    #                 inner_batch_labels = batch["edit_inner"]["labels"]
    #                 if not isinstance(inner_edit_outputs, torch.Tensor):
    #                     inner_edit_logits = inner_edit_outputs.logits
    #                 else:
    #                     inner_edit_logits = inner_edit_outputs

    #                 # if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
    #                 #     l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels)["nll"]
    #                 # else:
    #                 #     l_edit = self.model.edit_loss_fn(self.config, post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])["nll"]
    #                 # if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
    #                 #     l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels)["nll"]
    #                 # else:
    #                 #     l_image_edit = self.model.edit_loss_fn(self.config, post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])["nll"]               
                    
    #                 # Collect some useful metrics
    #                     # post_edit_dict = self.model.edit_loss_fn(post_edit_logits, post_batch_labels)
    #                     # l_edit = post_edit_dict["nll"]
    #                     # inner_edit_dict = self.model.edit_loss_fn(inner_edit_logits, inner_batch_labels)
    #                     # image_rephrase_edit_dict = self.model.edit_loss_fn(post_image_edit_logits, post_image_batch_labels)
    #                     # l_image_edit = image_rephrase_edit_dict["nll"]
                    
    #                 if post_edit_logits.shape[1] > post_batch_labels.shape[1]:
    #                     post_edit_dict = multiclass_log_probs(post_edit_logits, post_batch_labels)
    #                 else:
    #                     post_edit_dict = multiclass_log_probs(post_edit_logits, post_batch_labels[:, -post_edit_logits.shape[1]-1:])

    #                 if inner_edit_logits.shape[1] > inner_batch_labels.shape[1]:
    #                     inner_edit_dict = multiclass_log_probs(inner_edit_logits, inner_batch_labels)
    #                 else:
    #                     inner_edit_dict = multiclass_log_probs(inner_edit_logits, inner_batch_labels[:, -inner_edit_logits.shape[1]-1:])

    #                 if post_image_edit_logits.shape[1] > post_image_batch_labels.shape[1]:    
    #                     image_rephrase_edit_dict = multiclass_log_probs(post_image_edit_logits, post_image_batch_labels)
    #                 else:
    #                     image_rephrase_edit_dict = multiclass_log_probs(post_image_edit_logits, post_image_batch_labels[:, -post_image_edit_logits.shape[1]-1:])
                    
    #                 # post_base_outputs = self.alg.get_output(batch["loc"])
    #                 # if not isinstance(post_base_outputs, torch.Tensor):
    #                 #     post_base_logits = post_base_outputs.logits
    #                 #     kl_mask = post_base_outputs.attention_mask
    #                 # else:
    #                 #     post_base_logits = post_base_outputs
    #                 #     kl_mask = torch.ones(post_base_logits.shape[0], post_base_logits.shape[1]).to(post_base_logits.device)

    #                 post_image_base_outputs = self.alg.get_output(batch["loc_image"])
    #                 if not isinstance(post_image_base_outputs, torch.Tensor):
    #                     post_image_base_logits = post_image_base_outputs.logits
    #                     #kl_image_mask = post_image_base_outputs.attention_mask
    #                 else:
    #                     post_image_base_logits = post_image_base_outputs
    #                     #kl_image_mask = torch.ones(post_image_base_logits.shape[0], post_image_base_logits.shape[1]).to(base_image_logits.device)

    #                 #l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)
    #                 #l_image_loc = kl_loc_loss(base_image_logits.detach(), post_image_base_logits, mask=kl_image_mask)

    #             # if l_edit.isnan():
    #             #     print("l_edit is nan")
    #             #     print("input: ", batch["edit_outer"]['text_input'])
    #             # elif l_image_edit.isnan():
    #             #     print("l_image_edit is nan")
    #             #     print("input: ", batch["edit_outer_image"]['text_input'])
    #             # elif l_loc.isnan():
    #             #     print("l_loc is nan")
    #             #     print("input: ", batch["loc"]['text_input'])
    #             # elif l_image_loc.isnan():
    #             #     print("l_image_loc is nan")
    #             #     print("input: ", batch["loc_image"]['text_input'])

    #             # l_total_edit = self.config.cedit * l_edit + self.config.cloc * (l_loc+l_image_loc) + self.config.iedit * l_image_edit

    #             # if training and self.config.alg != 'ft':
    #             #     safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

    #             # Text locality
    #             #post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
    #             #base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

    #             # Image locality
    #             post_image_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_image_base_logits, dim=-1), k=10, dim=-1).indices
    #             base_image_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_image_logits, dim=-1), k=10, dim=-1).indices

    #             info_dict = {}
    #             # info_dict['loss/edit'] = l_edit.item()
    #             # info_dict['loss/image_edit'] = l_image_edit.item()
    #             # info_dict['loss/loc'] = l_loc.item()
    #             info_dict['edit/acc'] = post_edit_dict["acc"].item()
    #             info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
    #             info_dict['edit/prob'] = post_edit_dict["prob"].item()
    #             info_dict['inner/acc'] = inner_edit_dict["acc"].item()
    #             info_dict['image_rephrase/acc'] = image_rephrase_edit_dict["acc"].item()
    #             info_dict["time/edit"] = edit_time
    #             #info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
    #             info_dict["image_loc/acc"] = sum(post_image_base_logits_softmax_top_k.view(-1) == base_image_logits_softmax_top_k.view(-1))/post_image_base_logits_softmax_top_k.view(-1).shape[0]
    #             # l_base = torch.tensor(0.0)
    #             # l_total = l_total_edit + self.config.cbase * l_base

    #             # info_dict["loss/total"] = l_total.item()
    #             # info_dict["loss/total_edit"] = l_total_edit.item()
    #             # info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
    #             # info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
    #             # info_dict = {**info_dict, **model_info}
    #             print(info_dict)


    def run_edit(self):
        # --- editing start ---
        self.alg.enable_melo()
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
        revo=[]

        for i, batch in tqdm(enumerate(self.eval_loader)):
            LOG.info(f'-------------------------    Edit Batch {i} ----------------------------------')
            if n_edits < self.config.max_n_edits:
                n_edits += self.batch_size
                batch_history.append(batch)

                #To 
                with torch.no_grad():
                    self.alg.disable_melo()


                    ###测试一下vision_outputs
                    vision_outputs = self.alg.get_output(batch["edit_inner"]).vision_outputs.last_hidden_state[:,0,:]
                    re_vision_outputs = self.alg.get_output(batch["edit_outer_image"]).vision_outputs.last_hidden_state[:,0,:]
                    vo.append(vision_outputs)
                    revo.append(re_vision_outputs)
                    for iii in vo:
                        #print(torch.cdist(re_vision_outputs, iii, p=2, compute_mode='donot_use_mm_for_euclid_dist'))
                        print(torch.nn.functional.cosine_similarity(re_vision_outputs, iii,dim=1))
                    ###测试完毕

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


                # result=compute_multimodal_edit_results(self.alg,batch,self.processor)
                # print(result)
                
                # --- perform edit ---
                self.alg.enable_melo()
                edit_start = time()
                self.alg.edit(batch["edit_inner"])
                edit_time = time() - edit_start
                total_edit_time += edit_time

            # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    # Editing loss
                    result=compute_multimodal_edit_results(self.alg,batch,self.processor)
                    print(result)


                    post_base_outputs = self.alg.get_output(batch["loc"])
                    if not isinstance(post_base_outputs, torch.Tensor):
                        post_base_logits = post_base_outputs.logits
                    else:
                        post_base_logits = post_base_outputs
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

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
                    
                    LOG.info(f"Batch {i} after Editing: edit/acc: {info_dict['edit/acc']} || inner/rephrase_acc: {info_dict['inner/rephrase_acc']} || image_rephrase/acc: {info_dict['image_rephrase/acc']} || image_loc/acc: {info_dict['image_loc/acc']} || loc/acc: {info_dict['loc/acc']}")

                    if (i > 0 and n_edits % self.config.grace.metric_period == 0) or (i == len(self.eval_loader) - 1):
                        LOG.info(f'-------------------------    Eval all {n_edits} history edits----------------------------------')
                        averager = RunningStatAverager("val")

                        for k,iter in enumerate(batch_history):
                            result = compute_multimodal_edit_results(self.alg, iter , self.processor)
                            # Text locality
                            post_base_outputs = self.alg.get_output(iter["loc"])
                            if not isinstance(post_base_outputs, torch.Tensor):
                                post_base_logits = post_base_outputs.logits
                            else:
                                post_base_logits = post_base_outputs
                            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(loc_history[k]["loc"], dim=-1), k=1, dim=-1).indices

                            # Image locality
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


                # result=compute_multimodal_edit_results(self.alg,batch,self.processor)
                # print(result)
                
                # --- perform edit ---
                self.alg.enable_melo()
                edit_start = time()
                self.alg.edit(batch["edit_inner"])
                edit_time = time() - edit_start
                total_edit_time += edit_time

            # --- Compute and log metrics ---
                log_dict = {}
                with torch.no_grad():
                    # Editing loss
                    result=compute_multimodal_edit_results(self.alg,batch,self.processor)
                    print(result)


                    post_base_outputs = self.alg.get_output(batch["loc"])
                    if not isinstance(post_base_outputs, torch.Tensor):
                        post_base_logits = post_base_outputs.logits
                    else:
                        post_base_logits = post_base_outputs
                    post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                    base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

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
                    
                    LOG.info(f"Batch {i} after Editing: edit/acc: {info_dict['edit/acc']} || inner/rephrase_acc: {info_dict['inner/rephrase_acc']} || image_rephrase/acc: {info_dict['image_rephrase/acc']} || image_loc/acc: {info_dict['image_loc/acc']} || loc/acc: {info_dict['loc/acc']}")

                    if (i > 0 and n_edits % self.config.grace.metric_period == 0) or (i == len(self.eval_loader) - 1):
                        LOG.info(f'-------------------------    Eval all {n_edits} history edits----------------------------------')
                        averager = RunningStatAverager("val")

                        for k,iter in enumerate(batch_history):
                            result = compute_multimodal_edit_results(self.alg, iter , self.processor)
                            # Text locality
                            post_base_outputs = self.alg.get_output(iter["loc"])
                            if not isinstance(post_base_outputs, torch.Tensor):
                                post_base_logits = post_base_outputs.logits
                            else:
                                post_base_logits = post_base_outputs
                            post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
                            base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(loc_history[k]["loc"], dim=-1), k=1, dim=-1).indices

                            # Image locality
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
                        



