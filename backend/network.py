# network.py — PyTorch Network wrapper for ONNX/PT export

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, network):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.topology = network.topology.copy()

        for i in range(len(network.topology) - 1):
            layer = nn.Linear(network.topology[i], network.topology[i+1])
            # weights and biases from network
            with torch.no_grad():
                layer.weight.copy_(torch.tensor(network.W[i], dtype=torch.float32))
                layer.bias.copy_(torch.tensor(network.b[i], dtype=torch.float32))
            setattr(self, f'fc{i+1}', layer)
            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x