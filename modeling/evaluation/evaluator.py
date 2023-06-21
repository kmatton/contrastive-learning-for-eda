"""
Module for evaluating model.
"""
import time

import torch
from torch.utils.data.dataloader import DataLoader

from modeling.evaluation.eval_results import EvalResults
from modeling.model_runner import ModelRunner


class Evaluator(ModelRunner):
    """
    Base class for evaluating model
    """
    def __init__(self, model, model_args, multi_process_args, loss_args, logger, eval_dataset, eval_args):
        """
        :param model: model to use
        :param model_args: arguments passed into model
        :param multi_process_args: arguments related to multi-processing
        :param loss_args: arguments related to loss function
        :param logger: module used for logging progress
        :param eval_dataset: dataset ot use for model evaluation
        :param eval_args: arguments related to evaluation
        """
        super().__init__(model, model_args, multi_process_args, loss_args, logger)
        self.eval_dataset = eval_dataset
        self.eval_args = eval_args

    def evaluate(self):
        self.logger.info("Starting EVALUATE")
        eval_start_time = time.time()
        eval_dataloader = self._get_eval_dataloader()
        eval_results = self._eval_epoch(eval_dataloader)
        for key, val in eval_results.metrics.items():
            self.logger.info(f"{key}: {val}")
        eval_time = time.time() - eval_start_time
        eval_results.add_time_info(eval_time)
        self.logger.info(f"EVAL finished in {eval_time}")
        return eval_results

    def _eval_epoch(self, eval_dataloader):
        store_preds = self.eval_args.save_preds or self.eval_args.metrics is not None
        num_batches = len(eval_dataloader)
        num_samples = len(self.eval_dataset)
        eval_results = EvalResults(num_batches,
                                   num_samples,
                                   store_preds,
                                   self.model.output_dim,
                                   self.eval_dataset.y_dim,
                                   self.eval_dataset.y_var_type)
        self.model.eval()
        for step, batch in enumerate(eval_dataloader):
            loss, preds = self._eval_step(batch)
            eval_results.add_step_result(step, loss, preds, batch)
        eval_results.compute_mean_loss()
        if self.eval_args.metrics is not None:
            eval_results.compute_metrics(self.eval_args.metrics, normalize_preds=True)
        return eval_results

    def _eval_step(self, batch):
        with torch.no_grad():
            loss, preds = self.model(batch, return_preds=True)
            if self.multi_process_args.apply_data_parallel and self.multi_process_args.n_gpu > 1:
                loss = loss.mean()
        return loss, preds

    def _get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.eval_args.batch_size,
            shuffle=False,
            num_workers=self.multi_process_args.num_workers,
            pin_memory=True,
            drop_last=False
        )
