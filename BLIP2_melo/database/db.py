import torch
from database.tools import cos_sim, NO_LORA

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


    def add_cluster(self, new_key, new_value, new_edit_label, id):
        new_row = {'cluster_center': None, 'key_label': None, 'points': []}
        new_row['cluster_center'] = new_key.detach()
        new_row['key_label'] = new_edit_label
        new_row['points'].append(mem_point(new_key.detach(), new_value, id))

        self.table.append(new_row)

    def norm(self, dist):
        if self.config.image_encoder_name == 'clip':
            norm_res = (dist-(-0.0115))/(1.0-(-0.0115))
        elif self.config.image_encoder_name == 'vit':
            norm_res = (dist - (-0.2437)) / (1.0 - (-0.2437))
        elif self.config.image_encoder_name == 'dino':
            norm_res = (dist-(-0.4940))/(1.0-(-0.4940))
        else:
            raise ValueError("Unknown Encoder Name!")
        return norm_res

    def search_combine_database_v2(self, batch_query):
        dists = []
        for x in self.table:
            temp = cos_sim(batch_query, x['cluster_center'])
            dists.append(temp)
        dists = torch.stack(dists).view(-1, len(batch_query)).T
        return self.norm(dists)



class VecDB:
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

    def add_cluster(self, new_key, new_value, new_edit_label, id):
        new_row = {'cluster_center': None, 'key_label': None, 'points': []}
        new_row['cluster_center'] = new_key.detach()
        new_row['key_label'] = new_edit_label
        new_row['points'].append(mem_point(new_key.detach(), new_value, id))
        self.table.append(new_row)
        self.conflict_num += 1


    def norm(self, dist):
        return (dist - (0.1047)) / (1.0000 - 0.0919)


    def search_combine_database_v2(self, batch_query_vision):
        dists = []
        for x in self.table:
            temp = cos_sim(batch_query_vision, x['cluster_center'])
            dists.append(temp)
        dists = torch.stack(dists).view(-1, len(batch_query_vision)).T
        return self.norm(dists)

    def search_vision_text_cluster(self, text_dists, vision_dists):
        lora_mapping_block = []
        combine_dists = 2 - (text_dists + vision_dists)
        values, indices_list = torch.topk(combine_dists, k=1, largest=False)
        for value, id in zip(values, indices_list):
            if value[0] > 0.35:
                lora_mapping_block.append(NO_LORA)
            else:
                lora_mapping_block.append(self.table[id[0]]['points'][0].get_value())
        return lora_mapping_block






