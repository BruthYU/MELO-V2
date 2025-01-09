import torch
from utils import *
import logging

LOG = logging.getLogger(__name__)

'''Multimodal Metrics
'''


def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target, ]
    if isinstance(prompts, str):
        prompts = [prompts, ]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]

    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt, )) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]

    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len
    }
    return ret


def compute_multimodal_edit_quality(alg, router, batch):
    lora_block_mapping = router.get_lora_mapping(batch)

    '''Inference'''
    with torch.no_grad():
        outputs = alg.get_output(batch, lora_block_mapping)

        # LOG.info(f"---[Metric outputs]----")
        # LOG.info(outputs.logits.shape)
        # print(outputs.logits[:, :, 15])

        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()
        targ = batch["labels"].cpu()

    if logits.dim() == 3:
        logits = logits[:, :-1]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != 1
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()

    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    return acc, pred_ids.numpy()


def compute_multimodal_edit_results(alg, router, batch):
    ret = {}
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(alg, router, batch["edit_inner"])

    if "edit_outer" in batch.keys():
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(alg, router, batch["edit_outer"])

    if "edit_outer_image" in batch.keys():
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(alg, router, batch["edit_outer_image"])

    return ret




