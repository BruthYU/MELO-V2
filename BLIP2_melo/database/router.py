import torch
from torchvision import transforms
import torch.nn.functional as F
from database.db import *
from database.tools import *

class Router:
    def __init__(self, config):
        self.config = config
        self.image_processor, self.image_encoder = self.init_image_encoder()
        self.text_encoder = self.init_flag_embedding()
        self.VisionVecDB = VisionVecDB(config)
        self.VecDB = VecDB(config)
        self.memory_num = 0
        self.block_id = 0


    def init_image_encoder(self):
        if self.config.image_encoder_name == 'dino':
            processor = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])
            encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
            encoder = encoder.to(self.config.device)

        elif self.config.image_encoder_name == 'clip':
            from transformers import CLIPProcessor, CLIPModel
            encoder = CLIPModel.from_pretrained("/home/hy/Yjh/MELO-master/basemodel/vit")
            processor = CLIPProcessor.from_pretrained("/home/hy/Yjh/MELO-master/basemodel/vit")
        elif self.config.image_encoder_name == 'vit':
            from transformers import ViTFeatureExtractor, AutoModel
            processor = ViTFeatureExtractor.from_pretrained('/home/hy/Yjh/MELO-master/basemodel/VIT')
            encoder = AutoModel.from_pretrained('/home/hy/Yjh/MELO-master/basemodel/VIT')
        else:
            raise ValueError("Unknown Encoder Name!")

        return processor, encoder

    def init_flag_embedding(self):
        from FlagEmbedding import FlagModel
        encoder = FlagModel('/home/hy/Yjh/MELO-master/basemodel/flag-embedding/',
                            query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        return encoder

    def batch_embed(self, batch):
        if self.config.task=='vqa' :
            image, text_input = batch["ori_image"], batch["prompt_ids"]
        else:
            image, text_input = batch["ori_image"], batch["text_input"]

        # Text Embedding of Query
        batch_query = [torch.from_numpy(self.text_encoder.encode(x)).unsqueeze(0).float() for x in text_input]
        batch_query = torch.cat(batch_query,dim=0)

        # Image Embedding of Query
        batch_query_vision = None
        # No Image for Text Locality
        if image is not None:
            if self.config.image_encoder_name == 'dino':
                image_preprocess = [self.image_processor(x).unsqueeze(0).to("cuda")
                                    for x in image]
                image_preprocess = torch.cat(image_preprocess,dim=0)
                batch_query_vision = self.image_encoder(image_preprocess)

            elif self.config.image_encoder_name == 'clip':
                image_preprocess = [self.image_processor(images= x, return_tensors="pt", padding=True)["pixel_values"]
                                    for x in image]
                image_preprocess  = torch.cat(image_preprocess,dim=0)
                batch_query_vision= F.normalize(self.image_encoder.get_image_features(image_preprocess),p=2,dim=1)

            elif self.config.image_encoder_name == 'vit':
                image_preprocess = [self.image_processor(images=x, return_tensors="pt", padding=True)["pixel_values"]
                                    for x in image]
                image_preprocess = torch.cat(image_preprocess, dim=0)
                batch_query_vision = F.normalize(self.image_encoder(pixel_values=image_preprocess).pooler_output, p=2, dim=1)
        return batch_query, batch_query_vision

    def database_batch_add(self, batch_query, batch_query_vision):
        for idx, (query, query_vision) in enumerate(zip(batch_query, batch_query_vision)):
            self.VecDB.add_cluster(query.detach(), self.block_id, None, self.memory_num)
            self.VisionVecDB.add_cluster(query_vision.detach(), self.block_id, None, self.memory_num)
            self.memory_num += 1
        self.block_id += 1

    def search(self, batch_query, batch_query_vision):
        if batch_query_vision is None:
            text_dists = self.VecDB.search_combine_database_v2(batch_query)
            vision_dists = torch.zeros_like(text_dists)
        else:
            text_dists = self.VecDB.search_combine_database_v2(batch_query)
            vision_dists = self.VisionVecDB.search_combine_database_v2(batch_query_vision)
        lora_block_mapping = self.VecDB.search_vision_text_cluster(text_dists.to('cpu'),vision_dists.to('cpu'))
        return lora_block_mapping

    def get_lora_mapping(self, batch):
        batch_query, batch_query_vision = self.batch_embed(batch)
        return self.search(batch_query, batch_query_vision)
