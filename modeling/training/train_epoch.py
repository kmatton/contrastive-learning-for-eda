"""
Class for training a model for a single epoch (i.e., one run over the dataset).
"""

from modeling.model_runner import ModelRunner


class TrainEpoch(ModelRunner):
    """
    Base class for running one epoch of model training.
    """
    def __init__(self, model, model_args, multi_process_args, loss_args, logger, dataloader, opt, stop_checker,
                 train_hist):
        """
        :param model: model to train
        :param model_args: arguments passed into model
        :param multi_process_args: arguments related to multi-processing
        :param loss_args: arguments related to loss function
        :param logger: module used for logging progress
        :param dataloader: data to use
        :param opt: optimizer
        :param stop_checker: module for checking stopping criteria
        :param train_hist: module for storing training progress stats
        """
        super().__init__(model, model_args, multi_process_args, loss_args, logger)
        self.dataloader = dataloader
        self.opt = opt
        self.stop_checker = stop_checker
        self.train_hist = train_hist

    def run_train_epoch(self):
        self.logger.info(f"Starting epoch {self.train_hist.epoch}")
        self.model.train()
        for batch in self.dataloader:
            self.model.zero_grad()
            self._train_step(batch)
            self.opt.step()
            stop_training = self.stop_checker.check_stop(self.train_hist.epoch,
                                                         self.train_hist.step,
                                                         self.train_hist.sample_count)
            if stop_training:
                break
        mean_epoch_loss = self.train_hist.increment_epoch()
        self.logger.info(f"Mean loss for epoch {self.train_hist.epoch}: {mean_epoch_loss}")

    def _train_step(self, batch):
        loss = self.model(batch)
        if self.multi_process_args.apply_data_parallel and self.multi_process_args.n_gpu > 1:
            # aggregate loss from multiple GPUS
            loss = self._mp_aggregate_loss(loss)
        loss.backward()
        self.train_hist.increment_step(loss.detach(), len(batch))
