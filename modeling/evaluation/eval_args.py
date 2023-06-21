"""
Classes related to model evaluation arguments.
"""


class EvalArgs:
    """
    Base class for evaluation arguments.
    """
    def __init__(self, eval_epochs=5, metrics=None, batch_size=32, save_preds=False, splits=None):
        """
        :param eval_epochs: compute validation performance every this many epochs
        :param metrics: list of evaluation metrics to compute
        :param batch_size: batch size to use when evaluating model
        :param save_preds: if True, save model predictions on eval data
        :param splits: datasplits to evaluate on - default is just val
        """
        self.eval_epochs = eval_epochs
        self.metrics = metrics
        self.batch_size = batch_size
        self.save_preds = save_preds
        self.splits = ["val"] if splits is None else splits
