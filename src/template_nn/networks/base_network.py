import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def forward(self, x: torch.Tensor):
        pass

    def optimise(self):
        pass
