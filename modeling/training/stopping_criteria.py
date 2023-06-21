"""
Class for checking stopping criteria during training.
"""

import numpy as np


class StopChecker:
    """
    Base class for module that checks stopping criteria
    """
    def __init__(self, batch_size=32, best_metric="loss", best_metric_data="val", greater_is_better=False,
                 max_epochs=200, max_steps=None, max_samples=None, apply_early_stopping=True, es_threshold=0.001,
                 es_patience=3):
        """
        :param batch_size: number of examples to include in batch
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
        self.best_metric = best_metric
        self.best_metric_data = best_metric_data
        self.greater_is_better = greater_is_better
        self.operator = np.greater if greater_is_better else np.less
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.max_samples = max_samples
        self.apply_early_stopping = apply_early_stopping
        self.es_threshold = es_threshold
        self.es_patience = es_patience
        self.patience_counter = 0

    def check_early_stop(self, train_hist):
        """
        Check for early stopping based on performance.
        :param train_hist: class storing training hist (including best metric value seen so far)
        :return: whether to stop training
        """
        stop_training = False
        if self.best_metric_data == "train" and self.best_metric == "loss":
            metric_value = train_hist.train_loss_epochs[-1]
        else:
            metric_value = train_hist.eval_metrics[-1][f"{self.best_metric_data}_{self.best_metric}"]
        if train_hist.best_so_far is None or (
                self.operator(metric_value, train_hist.best_so_far)
                and abs(metric_value - train_hist.best_so_far) > self.es_threshold
        ):
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        if self.patience_counter >= self.es_patience:
            stop_training = True
        return stop_training

    def check_stop(self, epoch, step, sample_count):
        """
        Check whether to stop based on timing criteria.
        :param epoch: current epoch
        :param step: current step (global value, across epochs)
        :param sample_count: current sample count (global value, across epochs)
        :return: whether to stop
        """
        if self.max_epochs is not None and epoch >= self.max_epochs:
            return True
        if self.max_steps is not None and step >= self.max_steps:
            return True
        if self.max_samples is not None and sample_count >= self.max_samples:
            return True
        return False
