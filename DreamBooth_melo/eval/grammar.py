import torch
import os
os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
pass