"""
Utility functions to be used by neural networks.
"""

import math


def get_conv1d_output_dim(input_dim, padding, dilation, kernel_size, stride):
    return math.floor((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)


def get_maxpool1d_output_dim(input_dim, padding, dilation, kernel_size, stride):
    return math.floor((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
