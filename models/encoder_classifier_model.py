"""
Basic model that consists of encoder and classifier.
"""

import torch
import torch.nn as nn

from models.model import Model
from models.model_utils import get_neural_net


class EncoderClassifierModel(Model):
    """
    Base class for model that consists of encoder + classifier.
    """
    def __init__(self, encoder_name, encoder_args, classifier_name, classifier_args, encoder_state_dict=None,
                 freeze_encoder=False, classifier_state_dict=None, merge_wrists=False):
        """

        :param encoder_name: name of network to use for mapping from raw data to learned representations (h)
        :param encoder_args: arguments for encoder network
        :param classifier_name: name of network for classifier
        :param classifier_args: arguments for classifier network
        :param encoder_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        :param freeze_encoder: if True, freeze weights of encoder model
        :param classifier_state_dict (optional): path ot Pytorch state dict w/ weights for classifier model
        :param merge_wrists: if True, x is expected to be concatenation of L & R wrist embeddings
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_args = encoder_args
        self.classifier_name = classifier_name
        self.classifier_args = classifier_args
        self.encoder_state_dict = encoder_state_dict
        self.classifier_state_dict = classifier_state_dict
        self.merge_wrists = merge_wrists
        self.freeze_encoder = freeze_encoder
        # init encoder and classifier
        self.encoder_net = get_neural_net(self.encoder_name, self.encoder_args, self.encoder_state_dict,
                                          self.freeze_encoder)
        # get output dim of encoder and provide as input dim of classifier
        encoder_output_dim = self.encoder_net.output_dim
        self.classifier_args["input_dim"] = encoder_output_dim * 2 if self.merge_wrists else encoder_output_dim
        self.classifier_net = get_neural_net(self.classifier_name, self.classifier_args, self.classifier_state_dict)
        self.nn_dict = nn.ModuleDict({
            "encoder_net": self.encoder_net,
            "classifier_net": self.classifier_net
        })
        self.output_dim = self.classifier_net.output_dim

    def forward(self, batch, return_preds=False):
        self._check_loss_fn_set()  # confirm that loss function has been set
        # prepare input to NN
        batch = self._prepare_inputs(batch, prep_keys={'x'})
        x = batch['x']
        y = batch['y']
        if self.merge_wrists:
            half_len = self.encoder_net.input_dim
            x_l = x[:, :half_len]
            x_r = x[:, half_len:]
            h_l = self.encoder_net(x_l)
            h_r = self.encoder_net(x_r)
            h = torch.cat((h_l, h_r), dim=1)
        else:
            h = self.encoder_net(x)
        y_hat = self.classifier_net(h)
        y = self._prepare_targets(y)
        loss = self.loss_fn(y_hat, y)
        if return_preds:
            return loss, y_hat
        return loss
