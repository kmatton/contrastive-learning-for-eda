"""
Utility Functions for implementing models
"""

import torch

from models.neural_nets.linear_net import LinearNN
from models.neural_nets.CNN_encoder import CNNEncoder
from models.neural_nets.FCNN import FCNN
from models.neural_nets.small_CNN_encoder import SmallCNNEncoder


def get_neural_net(name, args, state_dict=None, freeze=False):
    """
    :param name: name of neural network to initialize
    :param args: arguments for neural net
    :param state_dict (optional): path to Pytorch state dict to use to initialize model
    :param freeze (optional): if True, will freeze weights of the network (so they won't be updated w/ backprop)
    :return: neural network pytorch module
    """
    if name == "LinearNN":
        net = LinearNN(**args)
    elif name == "CNNEncoder":
        net = CNNEncoder(**args)
    elif name == "SmallCNNEncoder":
        net = SmallCNNEncoder(**args)
    elif name == "FCNN":
        net = FCNN(**args)
    else:
        print(f"ERROR: Unrecognized network {name}")
        print("Exiting...")
        exit(1)
    if state_dict is not None:
        print(f"Loading {state_dict=}")
        net.load_state_dict(torch.load(state_dict, map_location='cpu'))
    if freeze:
        for param in net.parameters():
            param.requires_grad = False
    return net
