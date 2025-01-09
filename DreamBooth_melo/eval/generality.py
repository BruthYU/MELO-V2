import argparse
import os
from clip_eval import *
from dino_img_eval import *
from dataset import prompt_for_locality_test
from PIL import Image
from pathlib import Path
import json
import numpy as np
from dataset import *


def read_img(data_dir):
    img_path_list = list(Path(data_dir).iterdir())
    img_list = [Image.open(x) for x in img_path_list]
    return img_list

def CLIPEval_generality(alg, subject_list, identifier_list):
    data_dir = '../result/generality'
    instance_data_dir = os.path.join('../data/instances')
    alg_data_dir = os.path.join(data_dir, alg)


    evaluator = CLIP_ImageDirEvaluator('cuda')
    img_fidelity = []
    text_fidelity = []
    for idx, (iden, sub) in enumerate(zip(identifier_list, subject_list)):
        instance_name = sub.replace("_", " ")
        prompt_list = prompt_for_generality_test(iden, instance_name)
        for p_idx, prompt in enumerate(prompt_list):
            print(f'evaluating prompt: {prompt}')
            tmp_dir = f"{sub}"
            source = read_img(os.path.join(instance_data_dir, tmp_dir))
            target = read_img(os.path.join(alg_data_dir, tmp_dir, f'prompt_id_{p_idx}'))
            sim_img, sim_text = evaluator.evaluate(target, source, prompt)
            img_fidelity.append(sim_img)
            text_fidelity.append(sim_text)

    avg_img_fidelity = sum(img_fidelity) / len(img_fidelity)
    avg_text_fidelity = sum(text_fidelity) / len(text_fidelity)
    print(f'exp: generality, alg: {alg}, evaluator: CLIP, avg_img_fidelity: {avg_img_fidelity}, avg_text_fidelity: {avg_text_fidelity}')

def DINOEval_generality(alg, subject_list, identifier_list):
    data_dir = '../result/generality'
    instance_data_dir = os.path.join('../data/instances')
    alg_data_dir = os.path.join(data_dir, alg)

    evaluator = DINOEvaluator('cuda')
    img_fidelity = []
    for idx, (iden, sub) in enumerate(zip(identifier_list, subject_list)):
        instance_name = sub.replace("_", " ")
        prompt_list = prompt_for_generality_test(iden, instance_name)
        for p_idx, prompt in enumerate(prompt_list):
            print(f'evaluating prompt: {prompt}')
            tmp_dir = f"{sub}"
            source = read_img(os.path.join(instance_data_dir, tmp_dir))
            target = read_img(os.path.join(alg_data_dir, tmp_dir, f'prompt_id_{p_idx}'))
            sim_img = evaluator.img_to_img_similarity(target, source)
            img_fidelity.append(sim_img)

    avg_img_fidelity = sum(img_fidelity) / len(img_fidelity)
    print(f'exp: generality, alg: {alg}, evaluator: DINO, avg_img_fidelity: {avg_img_fidelity}')


    avg_img_fidelity = sum(img_fidelity) / len(img_fidelity)
    print(f'alg: {alg}, evaluator: DINO, avg_img_fidelity: {avg_img_fidelity}')






if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--alg', required=True, choices=['MELO', 'LoRA', 'FT'])
    parser.add_argument('--evaluator', required=True, choices=['CLIP','DINO'])
    args = parser.parse_args()

    with open('../data/data.json', 'r') as f:
        data_info = json.load(f)
    subject_list = list(data_info.keys())
    identifier_list = np.load("../data/rare_tokens/rare_tokens.npy")[:len(subject_list)]


    if args.evaluator == 'CLIP':
        CLIPEval_generality(args.alg, subject_list, identifier_list)
    else:
        DINOEval_generality(args.alg, subject_list, identifier_list)


