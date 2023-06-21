"""
Class for model training.
"""

import copy
import inspect
import json
import logging
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from modeling.evaluation.evaluator import Evaluator
from modeling.evaluation.eval_args import EvalArgs
from modeling.training.train_epoch import TrainEpoch
from modeling.training.train_history import TrainHistory
from modeling.training.args.logging_args import LoggingArgs
from modeling.training.args.loss_args import LossArguments
from modeling.training.args.opt_args import OptimizationArguments
from utils import is_jsonable, set_seed


class Trainer:
    """
    Base class for model trainer.
    """
    def __init__(self, train_args, multi_proc_args, model, train_dataset, val_dataset=None):
        """
        :param train_args: dictionary containing arguments related to model to training
            - should contain: seed, opt_args, loss_args, logging args, and eval args
        :param multi_proc_args: arguments related to multi-processing
        :param model: model to train
        :param train_dataset: dataset with training data - should work with pytorch dataloader
        :param val_dataset (optional): dataset with validation data - should work with pytorch dataloader
        """
        # setup args
        self.train_args_dict = train_args
        self.opt_args = self._get_opt_args()
        self.loss_args = self._get_loss_args()
        self.multi_proc_args = multi_proc_args
        self.logging_args = self._get_log_args()
        self.eval_args = self._get_eval_args()

        # setup logging
        self.logger = self._get_logger()

        # get device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # setup model
        self.model = model
        model_signature = inspect.signature(self.model.forward)
        self.model_input_args = list(model_signature.parameters.keys())
        self._setup_model()

        # setup data
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # set seed
        self.seed = self.train_args_dict["seed"]
        set_seed(self.seed)

        # init train hist
        self.train_hist = self._init_train_history()

    def train(self):
        """
        Train model.
        """
        self.logger.info("Starting TRAIN")
        train_start_time = time.time()

        # training setup - stopping checker & train epoch module
        stop_checker = self.opt_args.create_stop_checker()
        opt = self.opt_args.create_optimizer(self.model)
        train_dataloader = self._get_train_dataloader()
        train_epoch = TrainEpoch(
            self.model,
            self.model_input_args,
            self.multi_proc_args,
            self.loss_args,
            self.logger,
            train_dataloader,
            opt,
            stop_checker,
            self.train_hist
        )

        # run training
        self._run_train(train_epoch, stop_checker)
        self._log_train_completion(train_start_time)

        if self.opt_args.load_best_model_at_end:
            print("loading best model")
            self.model.load_state_dict(torch.load(os.path.join(self.logging_args.output_dir, "best_model.pt")))

    def _run_train(self, train_epoch, stop_checker):
        while True:
            train_epoch.run_train_epoch()
            # check if need to evaluate model, save model, and/or stop training
            stop_training = self._maybe_eval_save_stop(stop_checker)
            if stop_training:
                return

    def _log_train_completion(self, train_start_time):
        # log training completion and save train results
        self.logger.info(f"Stopping TRAIN after {self.train_hist.epoch} epochs, {self.train_hist.step} total steps")
        train_time = time.time() - train_start_time
        self.train_hist.add_train_time_info(train_time)
        self.logger.info("TRAIN finished in {:.2f} seconds".format(train_time))
        if self.logging_args.save_last_model:
            self.model.save(self.logging_args.output_dir, "final_model")
        self._save_train_hist("train_history.json")

    def evaluate(self, eval_dataset=None, eval_name="val", save_results=True):
        """
        Evaluate model. Can be called multiple times (e.g., with different datasets).
        :param eval_dataset: dataset to evaluate model with
        :param eval_name: name of evaluation, used in naming output result files
        :param save_results: if True, save eval results
        """
        # set dataset to default if not passed as argument
        if eval_dataset is None:
            eval_dataset = self.val_dataset
        evaluator = self._get_evaluator(eval_dataset)
        eval_results = evaluator.evaluate()
        if save_results:
            # save eval metrics
            self._save_eval_results(eval_results.metrics, f"{eval_name}_evaluation_metrics.json")
            if self.eval_args.save_preds:
                eval_results.save_preds(self.logging_args.output_dir)
        return eval_results

    def _get_evaluator(self, eval_dataset):
        evaluator = Evaluator(
            self.model,
            self.model_input_args,
            self.multi_proc_args,
            self.loss_args,
            self.logger,
            eval_dataset,
            self.eval_args
        )
        return evaluator

    def _maybe_eval_save_stop(self, stop_checker):
        stop_training = False
        do_eval = (self.val_dataset is not None and self.train_hist.epoch % self.eval_args.eval_epochs == 0)
        if do_eval:
            # evaluate model
            self.logger.info(f"Evaluating model after {self.train_hist.epoch} epochs")
            eval_metrics = dict()
            if "val" in self.eval_args.splits:
                val_eval_metrics = self.evaluate().metrics
                for k, v in val_eval_metrics.items():
                    eval_metrics[f"val_{k}"] = v
            if "train" in self.eval_args.splits:
                train_eval_metrics = self.evaluate(self.train_dataset, eval_name="train").metrics
                for k, v in train_eval_metrics.items():
                    eval_metrics[f"train_{k}"] = v
            self.train_hist.add_eval_result(eval_metrics, self.train_hist.epoch)
        if do_eval or \
                (self.opt_args.best_metric_data == "train" and self.opt_args.best_metric == "loss"):
            if stop_checker.apply_early_stopping:
                # check if should stop early (need to check before updating best so far)
                stop_training = stop_checker.check_early_stop(self.train_hist)
            if stop_training:
                self.logger.info(f"Stopping TRAIN early after {self.train_hist.epoch} epochs, "
                                 f"{self.train_hist.step} total steps")
            # check if latest epoch is best so far & update best so far based on this
            best_so_far = self.train_hist.check_update_best_so_far()
            if best_so_far:
                self.logger.info(f"{self.opt_args.best_metric_data}-{self.opt_args.best_metric} is best so far")
                # check if should save model
                if self.logging_args.save_best_model:
                    self.model.save(self.logging_args.output_dir, "best_model")
        if not stop_training:
            # check if should stop based on timing criteria
            stop_training = stop_checker.check_stop(self.train_hist.epoch,
                                                    self.train_hist.step,
                                                    self.train_hist.sample_count)
        return stop_training

    def _save_train_hist(self, file_name):
        train_hist_dict = copy.deepcopy(self.train_hist.__dict__)
        file_path = os.path.join(self.logging_args.output_dir, file_name)
        for k, v in train_hist_dict.items():
            # check if json serializable
            if not is_jsonable(v):
                train_hist_dict[k] = str(v)
        with open(file_path, 'w') as outfile:
            json.dump(train_hist_dict, outfile)

    def _save_eval_results(self, eval_metrics, file_name):
        file_path = os.path.join(self.logging_args.output_dir, file_name)
        with open(file_path, 'w') as outfile:
            json.dump(eval_metrics, outfile)

    def _get_logger(self):
        if self.logging_args.verbose:
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, stream=sys.stdout,
                                datefmt='%Y-%m-%d %H:%M:%S')
        else:
            output_file = os.path.join(self.logging_args.output_dir, "training.log")
            logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=output_file, filemode="w+",
                                level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        return logging.getLogger()

    def _get_opt_args(self):
        return OptimizationArguments(**self.train_args_dict["opt_args"])

    def _get_loss_args(self):
        return LossArguments(**self.train_args_dict["loss_args"])

    def _get_log_args(self):
        return LoggingArgs(**self.train_args_dict["log_args"])

    def _get_eval_args(self):
        args = {}
        if "eval_args" in self.train_args_dict:
            args = self.train_args_dict["eval_args"]
        return EvalArgs(**args)

    def _setup_model(self):
        # setup model --> note: we use loss function *within* model to make data parallel easier
        self.model.set_loss_fn(self.loss_args.get_loss_fn())
        if self.multi_proc_args.apply_data_parallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def _init_train_history(self):
        return TrainHistory(self.opt_args.best_metric, self.opt_args.best_metric_data, self.opt_args.greater_is_better)

    def _get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt_args.batch_size,
            shuffle=True,
            num_workers=self.multi_proc_args.num_workers,
            pin_memory=True,
            drop_last=True
        )
