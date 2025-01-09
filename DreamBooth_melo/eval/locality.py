import argparse
import os
from clip_eval import *
from dino_img_eval import *
from dataset import prompt_for_locality_test
from PIL import Image
from pathlib import Path

def read_img(data_dir):
    img_path_list = list(Path(data_dir).iterdir())
    img_list = [Image.open(x) for x in img_path_list]
    return img_list

def CLIPEval_locality(alg):
    data_dir = '../result/locality'
    base_data_dir = os.path.join(data_dir, 'base')
    alg_data_dir = os.path.join(data_dir, alg)
    prompt_list = prompt_for_locality_test()

    evaluator = CLIP_ImageDirEvaluator('cuda')
    img_fidelity = []
    text_fidelity = []
    for idx, prompt in enumerate(prompt_list):
        print(f'evaluating prompt: {prompt}')
        tmp_dir = f"prompt_id_{idx}"
        source = read_img(os.path.join(base_data_dir, tmp_dir))
        target = read_img(os.path.join(alg_data_dir, tmp_dir))
        sim_img, sim_text = evaluator.evaluate(target, source, prompt)
        img_fidelity.append(sim_img)
        text_fidelity.append(sim_text)

    avg_img_fidelity = sum(img_fidelity) / len(img_fidelity)
    avg_text_fidelity = sum(text_fidelity) / len(text_fidelity)
    print(f'alg: {alg}, evaluator: CLIP, avg_img_fidelity: {avg_img_fidelity}, avg_text_fidelity: {avg_text_fidelity}')

def DINOEval_locality(alg):
    data_dir = '../result/locality'
    base_data_dir = os.path.join(data_dir, 'base')
    alg_data_dir = os.path.join(data_dir, alg)
    prompt_list = prompt_for_locality_test()

    evaluator = DINOEvaluator('cuda')
    img_fidelity = []
    text_fidelity = []
    for idx, prompt in enumerate(prompt_list):
        print(f'evaluating prompt: {prompt}')
        tmp_dir = f"prompt_id_{idx}"
        source = read_img(os.path.join(base_data_dir, tmp_dir))
        target = read_img(os.path.join(alg_data_dir, tmp_dir))
        sim_img = evaluator.img_to_img_similarity(source, target)
        img_fidelity.append(sim_img)


    avg_img_fidelity = sum(img_fidelity) / len(img_fidelity)
    print(f'exp: locality, alg: {alg}, evaluator: DINO, avg_img_fidelity: {avg_img_fidelity}')






if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--alg', required=True, choices=['MELO', 'LoRA', 'FT'])
    parser.add_argument('--evaluator', required=True, choices=['CLIP','DINO'])
    args = parser.parse_args()

    if args.evaluator == 'CLIP':
        CLIPEval_locality(args.alg)
    else:
        DINOEval_locality(args.alg)


