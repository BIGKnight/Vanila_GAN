import torch.nn as nn
import torch

class Generator(torch.nn.Module):
    def __init__(self, in_dim):
        super(Generator, self).__init__()
        self.dense_1 = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(inplace=True))
        self.dense_2 = nn.Sequential(nn.Linear(128, 784), nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x