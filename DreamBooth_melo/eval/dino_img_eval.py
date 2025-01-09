import clip
import torch
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
import os
from PIL import Image
from pathlib import Path
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'


class DINOEvaluator(object):
    def __init__(self, device) -> None:
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        self.model = self.model.to(self.device)

        self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)



    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        images = [self.preprocess(x).to(self.device) for x in images]
        images = torch.stack(images)
        return self.model(images)



    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        matix = src_img_features @ gen_img_features.T

        return (src_img_features @ gen_img_features.T).mean()







if __name__ == '__main__':
    dir_evaluator = DINOEvaluator('cuda')
    src_dir = '../data/instances/cat'
    gen_dir = '../data/instances/cat'

    src_img_path_list = list(Path(src_dir).iterdir())
    src_img_list = [Image.open(x) for x in src_img_path_list]


    gen_img_path_list = list(Path(gen_dir).iterdir())
    gen_img_list = [Image.open(x) for x in gen_img_path_list]


    sim = dir_evaluator.img_to_img_similarity(src_img_list, gen_img_list)
    print(sim)




