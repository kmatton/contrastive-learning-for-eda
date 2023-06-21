"""
Class for running model on dataset (either training or inference).
"""


class ModelRunner:
    """
    Base class for running model on dataset.
    """
    def __init__(self, model, model_input_args, multi_process_args, loss_args, logger):
        """
        :param model: model to use
        :param model_input_args: arguments passed into model
        :param multi_process_args: arguments related to multi-processing
        :param loss_args: arguments related to loss function
        :param logger: module used for logging progress
        """
        self.model = model
        self.model_input_args = model_input_args
        self.multi_process_args = multi_process_args
        self.loss_args = loss_args
        self.logger = logger

    def _mp_aggregate_loss(self, loss):
        if self.loss_args.weight_samples:
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss
