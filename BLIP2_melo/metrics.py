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
        outputs = model.model_output(batch)
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



def compute_multimodal_edit_results(
    model,
    record,
    processor
):
    ret = {}
    model.get_image(record["edit_inner"])
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, record["edit_inner"],processor)
    
    if "edit_outer" in record.keys():
        model.get_image(record["edit_outer"])
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, record["edit_outer"],processor)

        
    if "edit_outer_image" in record.keys():
        model.get_image(record["edit_outer_image"])
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, record["edit_outer_image"],processor)


    return ret



