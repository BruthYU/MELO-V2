import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class downsample(nn.Module):
    def __init__(self, width, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm(width)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x

class Encoder(nn.Module):
    def __init__(self, context_length: int, width:int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.transformer = Transformer(width=width, layers=layers, heads = heads, attn_mask=attn_mask)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.down = nn.Sequential(OrderedDict([
            ("mlp_1", downsample(width,context_length, context_length // 2)),
            ("mlp_2", downsample(width,context_length // 2, context_length // 10)),
            ("mlp_3", downsample(width,context_length // 10, 1)),
        ]))
        self.initialize()

    def initialize(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self,x: torch.Tensor):
        x = self.transformer(x) + self.positional_embedding
        return self.down(x)





class BindedKeyCLIP(nn.Module):
    def __init__(self, context_length: int, width:int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.context_length = context_length
        self.width = width
        self.layers = layers
        self.heads = heads
        self.encoder = Encoder(context_length=context_length, width = width, layers = layers, heads = heads, attn_mask = attn_mask)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        pass

    def forward(self, pair_bind_1: torch.Tensor, pair_bind_2: torch.Tensor):
        bsz = pair_bind_1.shape[0]
        bind_features_1 = self.encoder(pair_bind_1).squeeze(dim=1)
        bind_features_2 = self.encoder(pair_bind_2).squeeze(dim=1)

        # normalized features
        bind_features_1 = bind_features_1 / bind_features_1.norm(dim=1, keepdim=True)
        bind_features_2 = bind_features_2 / bind_features_2.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_1_as_row = logit_scale * bind_features_1 @ bind_features_2.t()
        logits_2_as_row = logits_1_as_row.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_1_as_row, logits_2_as_row
    
    def encode(self, pair_bind_1):
        bsz = pair_bind_1.shape[0]
        bind_features_1 = self.encoder(pair_bind_1).squeeze(dim=1)

        return bind_features_1
    
    def clip_cos(self, bind_features_1, bind_features_2):
        # normalized features
        bind_features_2 = bind_features_2.unsqueeze(dim=0)
        bind_features_1 = bind_features_1 / bind_features_1.norm(dim=1, keepdim=True)
        bind_features_2 = bind_features_2 / bind_features_2.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_1_as_row = logit_scale * bind_features_1 @ bind_features_2.t()
        logits_2_as_row = logits_1_as_row.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_2_as_row


if __name__ == '__main__':
    context_length = 48
    width = 2096
    heads = width // 64
    keyCLIP = BindedKeyCLIP(context_length = 48, width=width, heads = heads, layers=2)
    pass










