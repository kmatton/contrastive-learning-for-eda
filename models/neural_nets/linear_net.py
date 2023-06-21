"""
Class for linear NN.
"""
import torch.nn as nn


class LinearNN(nn.Module):
    """
    Class for linear neural network
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x
