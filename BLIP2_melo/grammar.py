import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

model = Model()

module_list = [key for key, _ in model.named_modules()]

print(module_list)

conv1_module = model.get_submodule('conv1')
param_list = [x for x,_ in conv1_module.named_parameters()]

print(param_list)
conv1_weight = conv1_module.get_parameter("weight")
pass
