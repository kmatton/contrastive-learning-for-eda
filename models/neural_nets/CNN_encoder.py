"""
Class for basic CNN encoder.
"""

import torch.nn as nn
import torch

from models.neural_nets.nn_utils import get_conv1d_output_dim, get_maxpool1d_output_dim


class CNNEncoder(nn.Module):
    """
    CNN-based encoder.
    """
    def __init__(self, input_dim, dropout_prob=0.1, kernel_size=7, stride=3, output_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        self.kernel_size = kernel_size
        self.stride = stride
        self.cnn_layers = self._get_cnn_layers()
        self.cnn_out_dim = self._get_cnn_output_dim()
        self.linear_layer = nn.Sequential(
            nn.Linear(self.cnn_out_dim, self.output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        if len(x.shape) == 2:
            # make x 3-D
            x = torch.reshape(x, (x.shape[0], 1, -1))
        h = self.cnn_layers(x)
        # reshape so that representations are 1D
        h = torch.reshape(h, (h.shape[0], -1))
        # apply linear layer
        h = self.linear_layer(h)
        return h

    def _get_cnn_layers(self):
        cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=self.kernel_size, stride=self.stride),
            nn.BatchNorm1d(num_features=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv1d(4, 16, self.kernel_size, self.stride),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout_prob),
            nn.Conv1d(16, 32, self.kernel_size, self.stride),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(p=self.dropout_prob),
        )
        return cnn_layers

    def _get_cnn_output_dim(self):
        out1 = get_conv1d_output_dim(self.input_dim, 0, 1, self.kernel_size, self.stride)
        out1b = get_maxpool1d_output_dim(out1, 0, 1, 2, 2)
        out2 = get_conv1d_output_dim(out1b, 0, 1, self.kernel_size, self.stride)
        out2b = get_maxpool1d_output_dim(out2, 0, 1, 2, 2)
        out3 = get_conv1d_output_dim(out2b, 0, 1, self.kernel_size, self.stride)
        out3b = get_maxpool1d_output_dim(out3, 0, 1, 2, 2)
        final_out = 32 * out3b  # number of output channels * output dim of last layer
        return final_out
