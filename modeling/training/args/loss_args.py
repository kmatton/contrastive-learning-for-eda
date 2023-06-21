"""
Classes relating to loss function arguments
"""


from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from loss.contrastive_loss import NCELoss
from loss.cross_entropy_loss import MLMSampleWeightedCrossEntropy, WeightedCrossEntropy, \
    WeightedBinaryCrossEntropy


class LossArguments:
    """
    Base class for loss arguments.
    """
    def __init__(self, loss_fn_name="cross_entropy", weight_samples=False, vocab_size=None,
                 temperature=0.1):
        """
        :param loss_fn_name: name of loss function to use
        :param weight_samples: if True, use loss function that takes in sample weights and uses them
                               to weigh contribution to loss
        :param vocab_size: vocab size of LM model if using masked LM loss, otherwise N\A
        :param temperature: temperature parameter used for contrastive loss functions
        """
        self.loss_fn_name = loss_fn_name
        self.weight_samples = weight_samples
        self.vocab_size = vocab_size
        self.temperature = temperature

    def get_loss_fn(self):
        if self.loss_fn_name == "binary_cross_entropy":
            if self.weight_samples:
                return WeightedBinaryCrossEntropy()
            return BCEWithLogitsLoss()
        if self.loss_fn_name == "cross_entropy":
            if self.weight_samples:
                return WeightedCrossEntropy()
            else:
                return CrossEntropyLoss()
        if self.loss_fn_name == "mlm_sample_cross_entropy":
            assert self.vocab_size is not None, "need to specific vocab size when using MLM sample loss"
            if self.weight_samples:
                return MLMSampleWeightedCrossEntropy(self.vocab_size)
            return MLMSampleWeightedCrossEntropy(self.vocab_size)
        if self.loss_fn_name == "MSE":
            return MSELoss()
        if self.loss_fn_name == "NCE":
            return NCELoss(self.temperature)
        print(f"Unrecognized loss function {self.loss_fn_name}")
        print("Exiting...")
        exit(1)
