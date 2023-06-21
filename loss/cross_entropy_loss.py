"""
Custom cross entropy loss functions used when training models.
"""

import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


# Cross Entropy Losses

class MLMSampleWeightedCrossEntropy(nn.Module):
    """
    Take weighted sum of sample metrics.
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(self, logits, weights, labels):
        masked_lm_loss = self.loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        # get sum of metrics across tokens in each sample
        masked_lm_loss = masked_lm_loss.view(labels.shape).sum(dim=1)
        # get counts of non-ignored words in each sample
        non_ignore_counts = (labels != -100).sum(dim=1)
        non_ignore_counts[non_ignore_counts == 0] = 1
        # divide metrics by counts of words that contributed to the loss
        masked_lm_loss /= non_ignore_counts
        # compute weighted loss across all samples
        loss = (weights * masked_lm_loss).sum()
        return loss


class MLMMeanSampleCrossEntropy(nn.Module):
    """
    Take average loss per sample (rather than per-token).
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        masked_lm_loss = self.loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        # get sum of metrics across tokens in each sample
        masked_lm_loss = masked_lm_loss.view(labels.shape).sum(dim=1)
        # get counts of non-ignored words in each sample
        non_ignore_counts = (labels != -100).sum(dim=1)
        # divide metrics by counts of words that contributed to the loss
        masked_lm_loss /= non_ignore_counts
        return masked_lm_loss.mean()


class WeightedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels, weights=None):
        if len(labels.shape) > 1:
            losses = self.loss_fct(logits, labels.flatten())
        else:
            losses = self.loss_fct(logits, labels)
        if weights is not None:
            loss = (weights * losses).sum()
        else:
            loss = losses.mean()
        return loss


class WeightedBinaryCrossEntropy(nn.Module):
    """
    Weighted binary cross entropy wiht logits loss.
    """
    def __init__(self):
        super().__init__()
        self.loss_fct = BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, weights=None):
        losses = self.loss_fct(logits, labels)
        if weights is not None:
            loss = (weights * losses).sum()
        else:
            loss = losses.mean()
        return loss