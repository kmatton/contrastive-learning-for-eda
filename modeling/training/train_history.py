"""
Class to store training progress.
"""

import numpy as np
import torch


class TrainHistory:
    """
    Base class for storing results accumulated during training.
    """
    def __init__(self, best_metric="loss", best_metric_data="val", greater_is_better=False):
        """
        :param best_metric: metric to use when assessing which trained model is best and whether to stop early
                            -- note: if best_metric_data is training data, it assumed that this is "loss"
        :param best_metric_data: data split to use (train or val) when assessing what trained model is best
        :param greater_is_better: whether a greater or lesser metric value is better for `best_metric`
        """
        self.best_metric = best_metric
        self.best_metric_data = best_metric_data
        self.operator = np.greater if greater_is_better else np.less
        self.best_so_far = None
        self.best_so_far_epoch = -1
        self.train_loss_steps = [[]]  # a list per epoch
        self.train_loss_epochs = []
        self.eval_epochs = []  # store epochs during which eval occurred
        self.eval_metrics = []
        self.epoch = 0
        self.step = 0
        self.sample_count = 0
        self.train_time = None

    def increment_epoch(self):
        # convert  step losses to numpy (moves from GPU to CPU)
        self.train_loss_steps[self.epoch] = list(torch.tensor(self.train_loss_steps[self.epoch]).numpy())
        mean_epoch_loss = self._get_epoch_loss()
        self.train_loss_epochs.append(mean_epoch_loss)
        self.epoch += 1
        self.train_loss_steps.append([])
        return mean_epoch_loss

    def increment_step(self, loss, num_samples):
        """
        :param loss: loss or the current step
        :param num_samples: number of samples in the batch at the current step
        """
        self.train_loss_steps[self.epoch].append(loss)
        self.step += 1
        self.sample_count += num_samples

    def _get_epoch_loss(self):
        """
        Get mean loss across all steps in epoch.
        """
        step_losses = self.train_loss_steps[self.epoch]
        return np.mean(step_losses)

    def add_train_time_info(self, train_time):
        self.train_time = train_time

    def add_eval_result(self, metrics, epoch):
        """
        :param metrics: dict with metrics to add
        :param epoch: how many epochs the model has been trained for
        """
        self.eval_epochs.append(epoch)
        self.eval_metrics.append(metrics)

    def check_update_best_so_far(self):
        """
        Check if latest epoch results are the best so far, and update 'best_so_far' if so.
        Returns true if latest epoch results are best so far.
        """
        if self.best_metric_data == "train" and self.best_metric == "loss":
            latest_val = self.train_loss_epochs[-1]
        else:
            latest_val = self.eval_metrics[-1][f"{self.best_metric_data}_{self.best_metric}"]
        if self.best_so_far is None or self.operator(latest_val, self.best_so_far):
            self.best_so_far = latest_val
            return True
        return False
