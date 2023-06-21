"""
Base class for neural network models.
"""

import os

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Base class for neural network model.
    """
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loss_fn = None
        self.nn_dict = None

    def _check_loss_fn_set(self):
        if self.loss_fn is None:
            print("ERROR: need to set model loss function before calling forward!")
            print("Exiting...")
            exit(1)

    def save(self, output_dir, fname_prefix):
        # save whole model
        file_path = os.path.join(output_dir, fname_prefix+".pt")
        state_dict = self.state_dict()
        torch.save(state_dict, file_path)
        # save sub nets within model
        for nn_name, net in self.nn_dict.items():
            file_path = os.path.join(output_dir, fname_prefix+f"_{nn_name}.pt")
            state_dict = net.state_dict()
            torch.save(state_dict, file_path)

    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    def _prepare_inputs(self, inputs, prep_keys=None):
        """
        :param inputs: inputs to model to prepare for running on NN model
                       (i.e., put on the right device & convert to float)
        :param prep_keys: (iterable) keys of items to prepare for running through NN model
        :return: processed inputs
        """
        if prep_keys is None:
            prep_keys = set(inputs.keys())
        for k, v in inputs.items():
            if k in prep_keys:
                inputs[k] = v.to(device=self.device).float()
        return inputs

    def _prepare_targets(self, targets):
        """
        :param targets: prediction targets to prepare to use as argument to loss function
                       (i.e., put on the right device & convert to float if needed)
        :return: processed targets
        """
        if targets.dtype == torch.double:
            # convert to float
            targets = targets.float()
        # make sure target is 2D
        if len(targets.shape) == 1:
            targets = targets[:, None]
            # also need to convert to float if binary classification
            targets = targets.float()
        return targets.to(device=self.device)
