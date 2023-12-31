import torch
from torchvision import transforms


class mem_point:
    def __init__(self, key, value, id):
        self.key = key
        self.value = value
        self.id = id

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value

    def get_lora_id(self):
        return self.value

    def get_id(self):
        return self.id





class VisionVecDB:
    def __init__(self, config):
        self.config = config
        self.table = []
        self.forget_num = 0
        self.conflict_num = 0
        self.forget_keys = []

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        return self.table[item]

    def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def add_cluster(self, new_key, new_value, new_edit_label, id):
        new_row = {'cluster_center': None, 'radius': None, 'key_label': None, 'points': []}
        new_row['cluster_center'] = new_key.detach()
        new_row['radius'] = torch.tensor(self.config['init_vision_radius'], device=new_key.device).view(1)
        new_row['key_label'] = new_edit_label
        new_row['points'].append(mem_point(new_key.detach(), new_value, id))

        self.table.append(new_row)

    def norm(self, dist):
        if self.config.image_encoder_name == 'clip':
            norm_res = (dist-(-0.0115))/(1.0-(-0.0115))
        elif self.config.image_encoder_name == 'vit':
            norm_res = (dist - (-0.2437)) / (1.0 - (-0.2437))
        elif self.config.image_encoder_name == 'dino':
            norm_res = dist
        else:
            raise ValueError("Unknown Encoder Name!")
        return norm_res

    def search_combine_database_v2(self, batch_query):
        dists = []
        for x in self.table:
            temp = self.cos_sim(batch_query, x['cluster_center'])
            dists.append(temp)
        dists = torch.stack(dists).view(-1, len(batch_query)).T
        return self.norm(dists)




class Router:
    def __init__(self, config):
        self.config = config
        self.image_processor, self.image_encoder = self.init_image_encoder()
        self.text_encoder = self.init_flag_embedding()

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
            raise ValueError("Unknown Encoder Name !")

        return processor, encoder

    def init_flag_embedding(self):
        from FlagEmbedding import FlagModel
        encoder = FlagModel('/home/hy/Yjh/MELO-master/basemodel/flag-embedding/',
                            query_instruction_for_retrieval='Represent this sentence for searching relevant passages:')
        return encoder

