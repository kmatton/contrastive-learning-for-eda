"""
Basic model that consists of single network for classification.
"""

import torch.nn as nn

from models.model import Model
from models.model_utils import get_neural_net


class ClassifierModel(Model):
    """
    Class for model that consists of classifier network.
    """
    def __init__(self, classifier_name, classifier_args, classifier_state_dict=None):
        """
        :param classifier_name: name of network for classifier
        :param classifier_args: arguments for classifier network
        :param classifier_state_dict (optional): path ot Pytorch state dict w/ weights for classifier model
        """
        super().__init__()
        self.classifier_name = classifier_name
        self.classifier_args = classifier_args
        self.classifier_state_dict = classifier_state_dict
        # init network
        self.classifier_net = get_neural_net(classifier_name, classifier_args, classifier_state_dict)
        self.nn_dict = nn.ModuleDict({
            "classifier_net": self.classifier_net
        })
        self.output_dim = self.classifier_net.output_dim

    def forward(self, batch, return_preds=False):
        self._check_loss_fn_set()  # confirm that loss function has been set
        # prepare input to NN
        batch = self._prepare_inputs(batch, prep_keys={'x'})
        x = batch['x']
        y = batch['y']
        y_hat = self.classifier_net(x)
        y = self._prepare_targets(y)
        loss = self.loss_fn(y_hat, y)
        if return_preds:
            return loss, y_hat
        return loss
