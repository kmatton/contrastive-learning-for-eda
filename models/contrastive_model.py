"""
Basic architecture for contrastive learning network.
"""

import torch.nn as nn

from models.model import Model
from models.model_utils import get_neural_net


class ContrastiveModel(Model):
    """
    Class for basic contrastive learning model.
    """
    def __init__(self, encoder_name, encoder_args, encoder_state_dict=None, transform_head_name=None,
                 transform_head_args=None, transform_head_state_dict=None):
        """
        encoder_name: name of network to use for mapping from raw data to learned representations (h)
        encoder_args: arguments for encoder network
        encoder_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        transform_head_name (optional): name of network for mapping from learned representations (h) to metric embedding
        transform_head_args (optional): arguments for transform head network
        transform_head_state_dict (optional): path to Pytorch state dict that contains weights to initialize model with
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_args = encoder_args
        self.encoder_state_dict = encoder_state_dict
        self.transform_head_name = transform_head_name
        self.transform_head_args = transform_head_args
        self.transform_head_state_dict = transform_head_state_dict
        # init encoder and transformer head
        self.encoder_net = get_neural_net(self.encoder_name, self.encoder_args, self.encoder_state_dict)
        self.nn_dict = nn.ModuleDict({
            "encoder_net": self.encoder_net
        })
        self.output_dim = self.encoder_net.output_dim
        self.transform_head = None
        if self.transform_head_name is not None:
            assert self.transform_head_args, "Need to provide arguments for transform head"
            # input dim should be output dim of encoder
            self.transform_head_args["input_dim"] = self.encoder_net.output_dim
            self.transform_head = get_neural_net(self.transform_head_name, self.transform_head_args,
                                                 self.transform_head_state_dict)
            self.nn_dict["transform_head"] = self.transform_head
            self.output_dim = self.transform_head.output_dim

    def forward(self, batch):
        self._check_loss_fn_set()  # confirm that loss function has been set
        # prepare inputs to NNs
        batch = self._prepare_inputs(batch, prep_keys={'x_1', 'x_2'})
        # get two views of data
        x_1 = batch["x_1"]
        x_2 = batch["x_2"]
        h_1 = self.encoder_net(x_1)
        h_2 = self.encoder_net(x_2)
        if self.transform_head is not None:
            h_1 = self.transform_head(h_1)
            h_2 = self.transform_head(h_2)
        if type(self.loss_fn).__name__ == "NCELossTimeAware":
            loss = self.loss_fn(
                h_1,
                h_2,
                batch["subject_id_int"].to(device=self.device),
                batch["segment_start_milliseconds"].to(device=self.device),
            )
        else:
            loss = self.loss_fn(h_1, h_2)
        return loss
