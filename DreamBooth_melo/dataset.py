import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from PIL.ImageOps import exif_transpose
from utils import *


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            class_data_root = None,
            class_prompt = None,
            class_num = None,
            size = 512,
            center_crop = False,
            encoder_hidden_states = None,
            class_prompt_encoder_hidden_states = None,
            tokenizer_max_length = None
    ):
        self.size  = size,
        self.center_crop = False,
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states


        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example

def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def prompt_for_locality_test():
    prompt_list = [
        'Labrador in the style of Vermeer',
        'vincent van gogh, the starry',
        'Black and white street photography of a rainy night in New York, reflections on wet pavement',
        'Cubist painting of a bustling city market with different perspectives of people and stalls',
        'Underwater photography of a coral reef, with diverse marine life and a scuba diver for scale',
        'Pixel art of an 8-bit video game character running through a retro platformer level',
        'A cute cat with a bowtie',
        'A television in the bedroom',
        'johnny depp photo portrait, film noir style, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic',
        'Product shot of Nike shoes, 3d blender render, blue back ground',
        'Photo of construction site, workers, tilt shift effect, bokeh, Nikon',
        'A Red phone booth in the middle of empty street',
        'Google earth satellite image',
        'Sun rays coming through window',
        'Pencil painting of a beautiful girl'
        'Energetic Painting of a pool with flamingos',
        'Photo of a street in the 1970s, vivid color',
        'Cherry blossom in an empty park, by Thomas Kinkade',
        'beautiful landscape of lake and mountain,hyperrealistic,warm light,sunset,high definition,4K',
        'A poster of Game of Thrones',
        'An image of a polar bears in the year of 2024',
        'Full of formulas on a black board'
    ]
    return prompt_list


def prompt_for_generality_test(identifier, subject_name):
    prompt_list = [
        'a photo of {0} {1} in the jungle'.format(identifier, subject_name),
        'a {0} {1} in the snow'.format(identifier, subject_name),
        'a {0} {1} on the beach'.format(identifier, subject_name),
        'a {0} {1} on a cobblestone street'.format(identifier, subject_name),
        'a {0} {1} on top of pink fabric'.format(identifier, subject_name),
        'a {0} {1} on top of a wooden floor'.format(identifier, subject_name),
        'a {0} {1} with a city in the background'.format(identifier, subject_name),
        'a {0} {1} with a mountain in the background'.format(identifier, subject_name),
        'a {0} {1} with a blue house in the background'.format(identifier, subject_name),
        'a {0} {1} on top of a purple rug in a forest'.format(identifier, subject_name),
        'a {0} {1} with a wheat field in the background'.format(identifier, subject_name),
        'a {0} {1} with a tree and autumn leaves in the background'.format(identifier, subject_name),
        'a {0} {1} with the Eiffel Tower in the background'.format(identifier, subject_name),
        'a {0} {1} floating on top of water'.format(identifier, subject_name),
        'a {0} {1} floating in an ocean of milk'.format(identifier, subject_name),
        'a {0} {1} on top of green grass with sunflowers around it'.format(identifier, subject_name),
        'a {0} {1} on top of a mirror'.format(identifier, subject_name),
        'a {0} {1} on top of the sidewalk in a crowded street'.format(identifier, subject_name),
        'a photo of {0} {1} on top of a dirt road'.format(identifier, subject_name),
        'a photo of {0} {1} on top of a white rug'.format(identifier, subject_name),
        'a photo of red {0} {1}'.format(identifier, subject_name),
        'a photo of {0} {1} in the Acropolis'.format(identifier, subject_name),
        'a photo of shiny {0} {1}'.format(identifier, subject_name),
        'a photo of wet {0} {1}'.format(identifier, subject_name),
        'a photo of cube shaped {0} {1}'.format(identifier, subject_name)
    ]

    # Live subject prompt
    alive_prompt_list = [
        'a {0} {1} in the jungle'.format(identifier, subject_name),
        'a {0} {1} in the snow'.format(identifier, subject_name),
        'a {0} {1} on the beach'.format(identifier, subject_name),
        'a {0} {1} on a cobblestone street'.format(identifier, subject_name),
        'a {0} {1} on top of pink fabric'.format(identifier, subject_name),
        'a {0} {1} on top of a wooden floor'.format(identifier, subject_name),
        'a {0} {1} with a city in the background'.format(identifier, subject_name),
        'a {0} {1} with a mountain in the background'.format(identifier, subject_name),
        'a {0} {1} with a blue house in the background'.format(identifier, subject_name),
        'a {0} {1} on top of a purple rug in a forest'.format(identifier, subject_name),
        'a {0} {1} wearing a red hat'.format(identifier, subject_name),
        'a {0} {1} wearing a santa hat'.format(identifier, subject_name),
        'a {0} {1} wearing a rainbow scarf'.format(identifier, subject_name),
        'a {0} {1} wearing a black top hat and a monocle'.format(identifier, subject_name),
        'a {0} {1} in a chef outfit'.format(identifier, subject_name),
        'a {0} {1} in a firefighter outfit'.format(identifier, subject_name),
        'a {0} {1} in a police outfit'.format(identifier, subject_name),
        'a {0} {1} wearing pink glasses'.format(identifier, subject_name),
        'a {0} {1} wearing a yellow shirt'.format(identifier, subject_name),
        'a {0} {1} in a purple wizard outfit'.format(identifier, subject_name),
        'a red {0} {1}'.format(identifier, subject_name),
        'a purple {0} {1}'.format(identifier, subject_name),
        'a shiny {0} {1}'.format(identifier, subject_name),
        'a wet {0} {1}'.format(identifier, subject_name),
        'a cube shaped {0} {1}'.format(identifier, subject_name)
    ]
    if ('dog' in subject_name or 'cat' in subject_name) and 'backpack' not in subject_name:
        return alive_prompt_list

    return prompt_list

