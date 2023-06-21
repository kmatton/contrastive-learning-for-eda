"""
Class for basic CNN encoder.
"""

import torch.nn as nn

from models.neural_nets.CNN_encoder import CNNEncoder
from models.neural_nets.nn_utils import get_conv1d_output_dim, get_maxpool1d_output_dim


class SmallCNNEncoder(CNNEncoder):
    """
    CNN-based encoder (smaller than the default one)
    """
    def __init__(self, input_dim, dropout_prob=0.1, kernel_size=7, stride=3, output_dim=256):
        super().__init__(input_dim, dropout_prob, kernel_size, stride, output_dim)

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
        )
        return cnn_layers

    def _get_cnn_output_dim(self):
        out1 = get_conv1d_output_dim(self.input_dim, 0, 1, self.kernel_size, self.stride)
        out1b = get_maxpool1d_output_dim(out1, 0, 1, 2, 2)
        out2 = get_conv1d_output_dim(out1b, 0, 1, self.kernel_size, self.stride)
        out2b = get_maxpool1d_output_dim(out2, 0, 1, 2, 2)
        final_out = 16 * out2b  # number of output channels * output dim of last layer
        return final_out
