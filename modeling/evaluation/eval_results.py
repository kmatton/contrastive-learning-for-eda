"""
Classes for storing and processing evaluation results.
"""

import os

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, log_loss, confusion_matrix
from sklearn.preprocessing import LabelBinarizer

from utils import sigmoid


class EvalResults:
    def __init__(self, num_batches, num_samples, store_preds, y_pred_dim, y_true_dim, y_var_type):
        """
        :param num_batches: number of batches of data that will be used for evaluation
        :param num_samples: number of samples that will be used for evaluation
        :param store_preds: whether to store predicted values for each sample
        :param y_pred_dim: dimension of predicted values
        :param y_true_dim: dimension of ground truth labels
        :param y_var_type: type of label variable (continuous or categorical)
        """
        self.num_batches = num_batches
        self.num_samples = num_samples
        self.y_pred_dim = y_pred_dim
        self.y_true_dim = y_true_dim
        self.y_var_type = y_var_type
        # number of samples in each batch (useful because size of last batch may be different)
        self.batch_sizes = np.zeros(self.num_batches)
        self.losses = np.zeros(self.num_batches)  # losses per batch
        self.sample_counts = np.zeros(self.num_batches)
        self.sample_ids, self.y_pred, self.y_true = None, None, None
        self.store_preds = store_preds
        self.metrics = dict()
        if self.store_preds:
            self.sample_ids = None
            self.y_pred = np.zeros((self.num_samples, self.y_pred_dim))
            self.y_true = np.zeros((self.num_samples, self.y_true_dim))
            if self.y_var_type == "categorical" or self.y_var_type == "binary":
                self.y_true = self.y_true.astype(int)

    def add_all_preds(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def save_preds(self, output_dir):
        np.save(os.path.join(output_dir, "y_pred.npy"), self.y_pred)
        np.save(os.path.join(output_dir, "y_true.npy"), self.y_true)

    def add_step_result(self, step, loss, preds, batch):
        """
        :param step: current step in evaluation loop
        :param loss: loss for the batch of data
        :param preds: predicted values for the batch of data
        :param batch: the batch of data
        """
        batch_size = len(batch["ID"])
        self.batch_sizes[step] = batch_size
        self.losses[step] = loss.item()
        if self.store_preds:
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            _sample_ids = np.array(batch["ID"])
            self.sample_ids = _sample_ids if self.sample_ids is None else np.concatenate((self.sample_ids, _sample_ids))
            self.y_pred[start_idx:end_idx] = preds.cpu().numpy()
            _y_true = batch['y'].numpy()
            if len(_y_true.shape) == 1:  # reshape to match expected shape
                _y_true = _y_true[:, None]
            self.y_true[start_idx:end_idx] = _y_true

    def compute_mean_loss(self):
        total = np.dot(self.batch_sizes, self.losses)
        mean_loss = total / sum(self.batch_sizes)
        self.metrics["loss"] = mean_loss

    def add_time_info(self, eval_time):
        """
        :param eval_time: time it took to complete evaluation
        """
        self.metrics["time"] = eval_time

    def compute_metrics(self, metrics, normalize_preds=False):
        assert self.store_preds, "need to store prediction in order to compute metrics"
        for metric_name in metrics:
            if metric_name == "accuracy":
                assert self.y_var_type in ["categorical", "binary"], "can only compute accuracy for discrete vars"
                # convert y predictions from probs to categories
                _y_pred = _normalize_preds(self.y_pred, self.y_var_type) if normalize_preds else self.y_pred
                _y_pred = _convert_from_prob_to_class(_y_pred)
                self.metrics["accuracy"] = accuracy_score(self.y_true, _y_pred)
            elif metric_name == "AUC":
                assert self.y_var_type in ["categorical", "binary"], "can only compute AUC for discrete vals"
                _y_pred = _normalize_preds(self.y_pred, self.y_var_type) if normalize_preds else self.y_pred
                # if labels are binary and preds are 2-D, take prob's associated with label 1
                if self.y_var_type == "binary" and len(_y_pred.shape) > 1 and _y_pred.shape[1] == 2:
                    _y_pred = _y_pred[:, 1]
                # check if labels are all the same class --> if so AUC is undefined
                if len(np.unique(self.y_true)) == 1:
                    print("WARNING: only 1 class found in y_true, AUC is undefined")
                    return np.NAN
                # convert true labels from indices to one-hot encodings
                _y_true = _idx_to_one_hot(self.y_true)
                self.metrics["AUC"] = roc_auc_score(_y_true, _y_pred, multi_class='ovo')
            elif metric_name == "cross_entropy":
                assert self.y_var_type in ["categorical", "binary"], "can only compute cross entropy for discrete vars"
                _y_pred = _normalize_preds(self.y_pred, self.y_var_type) if normalize_preds else self.y_pred
                self.metrics["cross_entropy"] = log_loss(self.y_true, _y_pred)
            elif metric_name == "confusion_matrix":
                assert self.y_var_type in ["categorical", "binary"], "can only compute cross entropy for discrete vars"
                # convert y predictions from probs to categories
                _y_pred = _normalize_preds(self.y_pred, self.y_var_type) if normalize_preds else self.y_pred
                _y_pred = _convert_from_prob_to_class(_y_pred)
                self.metrics["confusion_matrix"] = confusion_matrix(self.y_true, _y_pred).tolist()
            elif metric_name == "MSE":
                assert self.y_var_type == "continuous", "can only compute MSE for continuous vars"
                self.metrics["MSE"] = mean_squared_error(self.y_true, self.y_pred)
            else:
                print(f"Unrecognized metric name {metric_name}")
                print("Exiting...")
                exit(1)


# helper functions
def _convert_from_prob_to_class(y_pred):
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        _y_pred = np.argmax(y_pred, axis=1)
    else:  # binary prediction
        _y_pred = y_pred > 0.5
    return _y_pred


def _normalize_preds(y_pred, y_var_type):
    """
    Used for when NN outputs logits rather than than sigmoid(logits)
    """
    if y_var_type == "categorical":
        return softmax(y_pred, axis=1)
    # otherwise will be binary
    return np.array([sigmoid(pred) for pred in y_pred])


def _idx_to_one_hot(y):
    lb = LabelBinarizer()
    return lb.fit_transform(y)
