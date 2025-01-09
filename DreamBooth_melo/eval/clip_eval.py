import clip
import torch
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
import os
from PIL import Image
from pathlib import Path
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'


class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        images = [self.clip_preprocess(x).to(self.device) for x in images]
        images = torch.stack(images)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

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

    def txt_to_img_similarity(self, text, generated_images):
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


class LDMCLIPEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, ldm_model, src_images, target_text, n_samples=64, n_steps=50):
        sampler = DDIMSampler(ldm_model)

        samples_per_batch = 8
        n_batches = n_samples // samples_per_batch

        # generate samples
        all_samples = list()
        with torch.no_grad():
            with ldm_model.ema_scope():
                uc = ldm_model.get_learned_conditioning(samples_per_batch * [""])

                for batch in range(n_batches):
                    c = ldm_model.get_learned_conditioning(samples_per_batch * [target_text])
                    shape = [4, 256 // 8, 256 // 8]
                    samples_ddim, _ = sampler.sample(S=n_steps,
                                                     conditioning=c,
                                                     batch_size=samples_per_batch,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=5.0,
                                                     unconditional_conditioning=uc,
                                                     eta=0.0)

                    x_samples_ddim = ldm_model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)

                    all_samples.append(x_samples_ddim)

        all_samples = torch.cat(all_samples, axis=0)

        sim_samples_to_img = self.img_to_img_similarity(src_images, all_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), all_samples)

        return sim_samples_to_img, sim_samples_to_text


class CLIP_ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text):
        sim_samples_to_img = self.img_to_img_similarity(src_images, gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)

        return sim_samples_to_img, sim_samples_to_text


if __name__ == '__main__':
    dir_evaluator = CLIP_ImageDirEvaluator('cuda')
    src_dir = '/data/instances/cat'
    gen_dir = '/data/instances/cat'

    src_img_path_list = list(Path(src_dir).iterdir())
    src_img_list = [Image.open(x) for x in src_img_path_list]


    gen_img_path_list = list(Path(gen_dir).iterdir())
    gen_img_list = [Image.open(x) for x in gen_img_path_list]


    sim = dir_evaluator.img_to_img_similarity(src_img_list, gen_img_list)
    print(sim)




