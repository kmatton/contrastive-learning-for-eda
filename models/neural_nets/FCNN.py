"""
Class for fully connected neural net
"""

import torch.nn as nn


class FCNN(nn.Module):
    """
    Class for fully connected neural net.
    """
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        layers = []
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            layers.append(nn.ReLU)
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
