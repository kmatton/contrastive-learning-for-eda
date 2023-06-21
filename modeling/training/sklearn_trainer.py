"""
Class for training sklearn model.
"""

from joblib import dump
import json
import logging
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from modeling.evaluation.eval_results import EvalResults
from modeling.training.train_utils import set_seed


class SKLearnTrainer:
    """
    Class for training sklearn model
    """
    def __init__(self, train_args, multi_proc_args, model, train_dataset, val_dataset=None):
        """
        :param train_args: dictionary containing arguments related to model to training
            - should contain: seed, log_args, train_metrics, eval_metrics
        :param multi_proc_args: arguments related to multi-processing
        :param model: model to train (should have fit and predict functions)
        :param train_dataset: dataset with training data (should have data_df member variable)
        :param val_dataset (optional): dataset with validation data (should have data_df member variable)
        """
        # setup args
        self.train_args_dict = train_args
        self.multi_proc_args = multi_proc_args
        self.train_metrics = self.train_args_dict["train_metrics"]
        self.eval_metrics = self.train_args_dict["eval_metrics"]

        # setup logging
        self.verbose = self.train_args_dict["log_args"]["verbose"]
        self.output_dir = self.train_args_dict["log_args"]["output_dir"]
        self.logger = self._get_logger()

        # setup model
        self.scaler = StandardScaler()
        self.model = model

        # setup data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # set seed
        self.seed = self.train_args_dict["seed"]
        set_seed(self.seed)

    def train(self):
        """
        Train model.
        """
        self.logger.info("Starting TRAIN")
        train_start_time = time.time()
        X = np.array(self.train_dataset.data_df["x"].tolist())
        y = self.train_dataset.data_df['y'].to_numpy()
        # apply z normalization to X
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        # fit model with data
        self.model.fit(X_scaled, y)
        self._log_train_completion(train_start_time)
        # get train performance
        y_pred = self.model.predict_proba(X_scaled)
        y_pred_dim = 1 if len(y_pred.shape) == 1 else y_pred.shape[1]
        train_results = EvalResults(
            num_batches=1,
            num_samples=len(X),
            store_preds=True,
            y_pred_dim=y_pred_dim,
            y_true_dim=self.train_dataset.y_dim,
            y_var_type=self.train_dataset.y_var_type
        )
        train_results.add_all_preds(y_pred, y)
        train_results.compute_metrics(self.train_metrics)
        # save train results
        self._save_eval_results(train_results.metrics, "train_metrics.json")

    def _log_train_completion(self, train_start_time):
        # log training completion and save train results
        train_time = time.time() - train_start_time
        self.logger.info("TRAIN finished in {:.2f} seconds".format(train_time))
        # save model
        dump(self.model, os.path.join(self.output_dir, "final_model.joblib"))

    def evaluate(self, eval_dataset=None, return_preds=False, eval_name="val"):
        """
        Evaluate model. Can be called multiple times (e.g., with different datasets).
        :param eval_dataset: dataset to evaluate model with
        :param return_preds: whether to return predictions made by model (vs. just evaluation metrics)
        :param eval_name: name of evaluation, used in naming output result files
        """
        # set dataset to default if not passed as argument
        if eval_dataset is None:
            eval_dataset = self.val_dataset
        X = np.array(eval_dataset.data_df["x"].tolist())
        X_scaled = self.scaler.transform(X)
        y = eval_dataset.data_df['y'].to_numpy()
        y_pred = self.model.predict_proba(X_scaled)
        y_pred_dim = 1 if len(y_pred.shape) == 1 else y_pred.shape[1]
        eval_results = EvalResults(
            num_batches=1,
            num_samples=len(X),
            store_preds=True,
            y_pred_dim=y_pred_dim,
            y_true_dim=eval_dataset.y_dim,
            y_var_type=eval_dataset.y_var_type
        )
        eval_results.add_all_preds(y_pred, y)
        eval_results.compute_metrics(self.eval_metrics)
        # save eval metrics
        self._save_eval_results(eval_results.metrics, f"{eval_name}_evaluation_metrics.json")
        return eval_results

    def _save_eval_results(self, eval_metrics, file_name):
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, 'w') as outfile:
            json.dump(eval_metrics, outfile)

    def _get_logger(self):
        if self.verbose:
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, stream=sys.stdout,
                                datefmt='%Y-%m-%d %H:%M:%S')
        else:
            output_file = os.path.join(self.output_dir, "training.log")
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=output_file, filemode="w+",
                                level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger()
