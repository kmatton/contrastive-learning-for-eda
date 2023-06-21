"""
Classes related to model optimization.
"""

import torch.optim as optim

from modeling.training.stopping_criteria import StopChecker


class OptimizationArguments:
    """
    Base class for handling optimization arguments for a single model.
    """
    def __init__(self, batch_size=32, optimizer_type="Adam", learning_rate=1e-02, weight_decay=0, momentum=0,
                 adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-08, load_best_model_at_end=True,
                 best_metric="loss", best_metric_data="val", greater_is_better=False, max_epochs=200, max_steps=None,
                 max_samples=None, apply_early_stopping=True, es_threshold=0.001, es_patience=3):
        """
        :param batch_size: number of examples to include in batch
        :param optimizer_type: type of optimizer to use
        :param learning_rate: initial learning rate
        :param weight_decay: weight decay to apply
        :param momentum: momentum to use if using SGD
        :param adam_beta1: beta 1 param of Adam optimizer
        :param adam_beta2: beta 2 param of Adam optimizer
        :param adam_epsilon: epsilon param of Adam optimizer
        :param load_best_model_at_end: whether to load the best model (from any epoch during training) at the end of
                                       training
        :param best_metric: metric to use when assessing which trained model is best and whether to stop early
                            -- note: if best_metric_data is training data, it assumed that this is "loss"
        :param best_metric_data: data split to use (train or val) when assessing what trained model is best
        :param greater_is_better: whether a greater or lesser metric value is better for `best_metric`
        :param max_epochs: max number of training epochs
        :param max_steps: max number of training steps (NOTE: overrides max_epochs)
        :param max_samples: max number of training samples (NOTE: overrides max_epochs and max_steps)
        :param apply_early_stopping: apply if true
        :param es_threshold: threshold to determine when to stop training
        :param es_patience: # of epochs to wait to stop if no progress is made
        """
        self.batch_size = batch_size

        # arguments related to optimizer used
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

        # arguments related to loading best model and stopping criteria
        self.load_best_model_at_end = load_best_model_at_end
        self.best_metric = best_metric
        self.best_metric_data = best_metric_data
        self.greater_is_better = greater_is_better
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.max_samples = max_samples
        self.apply_early_stopping = apply_early_stopping
        self.es_threshold = es_threshold
        self.es_patience = es_patience

    def create_optimizer(self, model):
        """
        :return: optimizer for the specified parameters of type `optimizer type`
        """
        if self.optimizer_type == "SGD":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                             weight_decay=self.weight_decay)
        elif self.optimizer_type == "Adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, betas=(self.adam_beta1, self.adam_beta2),
                              eps=self.adam_epsilon, weight_decay=self.weight_decay)
        else:
            print("ERROR: unrecognized optimizer type {}".format(self.optimizer_type))
            exit(1)

    def create_stop_checker(self):
        stop_checker = StopChecker(self.batch_size, self.best_metric, self.best_metric_data, self.greater_is_better,
                                   self.max_epochs, self.max_steps, self.max_samples, self.apply_early_stopping,
                                   self.es_threshold, self.es_patience)
        return stop_checker
