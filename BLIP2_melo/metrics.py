import torch
from utils import *
import logging
LOG = logging.getLogger(__name__)


# DEPRECATED
def sent_success(pre_edit_probs, post_edit_probs, pos_mask, eps=torch.finfo(torch.float32).eps, batch_size=20):
    assert False, "No longer used"
    # content_score = post_edit_probs[pos_mask].prod() ** (1/pos_mask.sum()) / (pre_edit_probs[pos_mask]. + eps)
    post_pos_avg = post_edit_probs[pos_mask].prod() ** (1 / pos_mask.sum())
    pre_pos_avg = pre_edit_probs[pos_mask].prod() ** (1 / pos_mask.sum())
    content_score = post_pos_avg / (pre_pos_avg + eps)
    z_content = min(1., content_score)

    # compute z_sent through a weighting objective
    # normalized_probs = post_edit_probs / (post_edit_probs.sum() + eps)
    # balancing_factor = 0.5 * ((~pos_mask).float().sum() / pos_mask.float().sum() + 1)
    # z_sent_weight = balancing_factor * normalized_probs.dot(pos_mask.float())
    post_neg_avg = post_edit_probs[~pos_mask].prod() ** (1 / (~pos_mask).sum())
    neg_over_pos = post_neg_avg / (eps + post_pos_avg)
    z_sent_weight = 1 / (1 + neg_over_pos)

    # compute z_sent through a ranking objective
    batch_mask = pos_mask.view(-1, batch_size).long()
    sort_idxs = post_edit_probs.view(-1, batch_size).sort(-1, descending=True).indices
    ranked_mask = batch_mask.gather(1, sort_idxs)
    true_mask = batch_mask.sort(-1, descending=True).values
    z_sent_rank = (ranked_mask == true_mask).float().mean()

    # compute the final success scores
    weight_success = (z_content * z_sent_weight) ** 0.5
    rank_success = (z_content * z_sent_rank) ** 0.5

    correct_probs = post_edit_probs[pos_mask].mean()
    wrong_probs = post_edit_probs[~pos_mask].mean()

    return {
        "acc_weight": weight_success,
        "acc_rank": rank_success,
        "rank_score": z_sent_rank,
        "weight_score": z_sent_weight,
        "content_score": content_score,
        "post_edit_probs": post_edit_probs.sum(),
        "pre_edit_probs": pre_edit_probs.sum(),
        "correct_probs": correct_probs,
        "wrong_probs": wrong_probs
    }



# For zsRE and F-NLI
def retain_rate(pre_logits, post_logits, mask=None):
    if pre_logits.shape[-1] == 1:
        pre_logits = pre_logits.squeeze(-1)
    if post_logits.shape[-1] == 1:
        post_logits = post_logits.squeeze(-1)

    assert pre_logits.shape == post_logits.shape
    assert pre_logits.shape[0] == mask.shape[0]

    if pre_logits.dim() == 1:
        # binary classification
        pre_preds = pre_logits > 0
        post_preds = post_logits > 0
        retain = (pre_preds == post_preds).float().mean()
    elif pre_logits.dim() == 3:
        # sequence modeling
        pre_preds = pre_logits.argmax(-1)
        post_preds = post_logits.argmax(-1)
        match = (pre_preds == post_preds) * mask
        retain = (match.sum(-1) == mask.sum(-1)).float().mean()
    else:
        raise NotImplementedError

    return retain.item()


def is_acc_error(model, tokens):
    # Check whether or not the model's prediction for a batch element is correct
    labels = tokens["labels"]
    logits = model(**tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return labels != argmaxs


def Accuracy(alg, tokens):
    labels = tokens["labels"]
    new_tokens = {f"{k}": v for k, v in tokens.items() if k != "labels"}
    logits = alg.model(**new_tokens).logits
    probs = torch.softmax(logits, -1).squeeze()
    argmaxs = torch.argmax(probs, dim=-1).squeeze()
    return (labels == argmaxs).float().mean()


def is_qa_error(model, tokens):
    preds = model.generate(tokens["input_ids"], max_length=20).squeeze()  # Run model to get its predictions
    labels = tokens["labels"]  # [tokens["labels"] != -100]

    if (len(preds) != len(labels)) or ((preds == labels).sum() != len(preds)):
        return True
    else:
        return False


def PPL(alg, batch):
    input_ids = batch["input_ids"][:, :1024]  # .to(device)
    if "labels" not in batch:
        batch["labels"] = batch["input_ids"][:, :1024].clone()
    else:
        batch["labels"] = batch["labels"][:, :1024].clone()

    with torch.no_grad():
        #outputs = alg.model.model(input_ids=input_ids, labels=target_ids)
        outputs = alg.model(**batch)
        nll = outputs.loss

    ppl = torch.exp(nll)  # .clip(0, 100)
    return ppl



def F1_ACC(alg, batch):
    try:
        preds = alg.generate(batch["input_ids"], max_length=20).squeeze()
        f1 = F1(preds, batch, alg.model_tok)
        acc = ACC(preds, batch, alg.model_tok)
        return f1, acc
    except Exception as e:
        raise e

def F1(preds, batch, tok):
    try:
        f1_list = []
        for p, g in zip(preds,batch["labels"]):
            p = p[p !=  tok.pad_token_id].cpu().squeeze()
            g = g[g != -100].cpu().squeeze()  # -100 might be nonsense
            num_same = len(np.intersect1d(p, g))
            len_pred = len(p)
            len_gold = len(g)
            precision = num_same / len_pred
            recall = 1.0 * num_same / len_gold
            f1 = (2 * precision * recall) / (precision + recall)
            f1_list.append(f1)
    except:
        return 0.

    return sum(f1_list) / len(f1_list)


def ACC(preds, batch, tok):
    decode_preds = tok.batch_decode(preds,skip_special_tokens=True)
    gold_labels = batch['labels']
    gold_labels = gold_labels.masked_fill(gold_labels == -100,tok.pad_token_id)
    decode_labels = tok.batch_decode(gold_labels,skip_special_tokens=True)
    assert len(decode_labels) == len(decode_preds), "Lengths of decode_preds and decode_labels should be the same"
    count = 0.
    for pred,label in zip(decode_preds, decode_labels):
        if pred == label:
            count = count + 1
    return count/len(decode_preds)



###multimodal

def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt,)) for prompt in prompts]  
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len        
    } 
    return ret

def compute_multimodal_edit_quality(model, batch, processor):
    
    with torch.no_grad():
        outputs = model.get_output(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()
            #logits = outputs.vision_outputs.detach().cpu()     
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        #targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    #mask = targ != -100
    mask = targ != 1
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy()


# def compute_multimodal_edit_quality(model, batch, processor):
    
#     with torch.no_grad():
#         outputs = model.generate_output(batch)
#         targ = batch["labels"].cpu()
#         outputs = outputs[:, 1:].detach().cpu()
#         if targ.shape[1]<=outputs.shape[1]:
#             outputs = outputs[:, :targ.shape[1]]
#         else:
#             targ = targ[:, :outputs.shape[1]]
#     #mask = targ != -100
#     mask = targ != 1
#     targ[~mask] = 0
#     pred_ids = outputs.masked_fill(~mask, 0).detach().cpu()
#     correct = pred_ids == targ
#     correct = correct & mask
#     num_non_padding = mask.sum().float().item()
#     acc = correct.sum() / num_non_padding

#     print(acc)
    
#     return acc, pred_ids.numpy()



# def compute_multimodal_edit_quality(model, batch,processor):
#     with torch.no_grad():
#         outputs=model.generate_output(batch)

#     outputs=outputs[:, 1:]
#     labels = batch["labels"]

#     acc=0

#     if outputs.shape[1]<=labels.shape[1]:
#         for i,iter in enumerate(outputs):
#             count=0
#             sum=0
#             for j,num in enumerate(iter):
#                 if num != 1 and num == labels[i][j]:
#                     count=count+1
#             for k in labels[i]:
#                 if k!=1:
#                     sum=sum+1
#             acc = acc + count/sum
#     else:
#         for i,iter in enumerate(labels):
#             count=0
#             sum=0
#             for j,num in enumerate(iter):
#                 if num != 1 and num == outputs[i][j]:
#                     count=count+1
#             for k in labels[i]:
#                 if k!=1:
#                     sum=sum+1
#             acc = acc + count/sum

#     return acc/outputs.shape[0]

    # decode_preds = processor.batch_decode(outputs,skip_special_tokens=True)

    # true_labels = batch["labels"].masked_fill(
    #             batch["labels"] == 1, -100
    #         )

    # decode_labels = processor.batch_decode(batch["labels"],skip_special_tokens=True)
    # assert len(decode_labels) == len(decode_preds), "Lengths of decode_preds and decode_labels should be the same"
    # count = 0.
    # for pred,label in zip(decode_preds, decode_labels):
    #     if pred == label:
    #         count = count + 1
    # return count/len(decode_preds)
    
    # # with torch.no_grad():
    # #     outputs = model.get_output(batch)
    # #     if isinstance(outputs, torch.Tensor):
    # #         logits = outputs.detach().cpu()
    # #     else:
    # #         logits = outputs.logits.detach().cpu()    
    # #     #targ = outputs.labels.detach().cpu()
    # #     targ = labels.cpu()
    # # if logits.dim() == 3:
    # #     logits = logits[:, :-1]
    # #     #targ = targ[:, 1:]
    # #     logits = logits[:, -targ.shape[1]:]
    # # mask = targ != -100
    # # targ[~mask] = 0
    # # pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    # # correct = pred_ids == targ
    # # correct = correct & mask
    # # num_non_padding = mask.sum().float().item()
    # # acc = correct.sum() / num_non_padding
    
    # return acc, bcc






# def compute_multimodal_edit_results(
#     model,
#     record,
#     processor
# ):
#     """
#     Given a rewritten model, computes generalization and specificity metrics for
#     the desired rewrite (passed in via the CounterFact dataset record). Returns a
#     dictionary containing those metrics.

#     :param model: Rewritten model
#     :param tok: Tokenizer
#     :param record: CounterFact dataset record
#     :paran snips: ???
#     :param vec: ???
#     :return: Dictionary containing rewriting metrics
#     """
#     ret = {}
#     # First, unpack rewrite evaluation record.
    
#     # target = record["target"]
#     # rewrite_prompts = record["prompt"]
#     # image = record["image"]
    
#     # edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
#     model.set_grace_store_mode(mode="ori")
#     ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, record["edit_inner"],processor)
#     model.set_grace_store_mode(mode= None)
    
#     if "edit_outer" in record.keys():
#         # rephrase_prompts = record["rephrase_prompt"]
#         #edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
#         model.set_grace_store_mode(mode="text_rephrase")
#         ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, record["edit_outer"],processor)
#         model.set_grace_store_mode(mode= None)
        
#     if "edit_outer_image" in record.keys():
#         #rephrase_image = record["image_rephrase"]
#         #edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
#         model.set_grace_store_mode(mode="image_rephrase") 
#         ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, record["edit_outer_image"],processor)
#         model.set_grace_store_mode(mode= None)


#     return ret


def compute_multimodal_edit_results(
    model,
    record,
    processor
):
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    ret = {}
    # First, unpack rewrite evaluation record.
    
    # target = record["target"]
    # rewrite_prompts = record["prompt"]
    # image = record["image"]
    
    # edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    model.get_image(record["edit_inner"])
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, record["edit_inner"],processor)
    
    if "edit_outer" in record.keys():
        # rephrase_prompts = record["rephrase_prompt"]
        #edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        model.get_image(record["edit_outer"])
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, record["edit_outer"],processor)

        
    if "edit_outer_image" in record.keys():
        #rephrase_image = record["image_rephrase"]
        #edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
        model.get_image(record["edit_outer_image"])
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, record["edit_outer_image"],processor)

    # if 'locality_prompt' in record.keys():
    #     locality_prompt = record["locality_prompt"]
    #     locality_ground_truth = record["locality_ground_truth"]
    #     #locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
    #     _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    # if 'loc_image' in record.keys():
    #     # m_loc_prompt = record["multimodal_locality_prompt"]
    #     # m_loc_ground_truth = record["multimodal_locality_ground_truth"]
    #     # m_loc_image = record["multimodal_locality_image"]
    #     #m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
    #     ret['multimodal_locality_acc'], ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, record["loc_image"],processor)
    # # Form a list of lists of prefixes to test.

    return ret



