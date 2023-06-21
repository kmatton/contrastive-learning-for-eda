"""
Loss functions for contrastive learning.
"""
import torch.nn as nn
import torch


class NCELoss(nn.Module):
    """
    Noise Contrastive Estimation Loss
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_v1, embeddings_v2, key_ids=None):
        """
        embeddings_v1: view 1 of learned representations (h) for each sample, potentially transformed
        embeddings_v2: view 2 of learned representations (h) for each sample, transformed
        key_ids (optional): ids denoting which samples should be considered the same vs different for
                 contrastive learning. If not provided, will assume each instance is the same only to
                 itself (i.e., embeddings_v1[i] and embeddings_v2[i] are pos pairs)
        """
        # create similarity matrix for all pairs of embeddings
        norm1 = embeddings_v1.norm(dim=1).unsqueeze(0)
        norm2 = embeddings_v2.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(embeddings_v1, embeddings_v2.transpose(0, 1))
        norm_matrix = torch.mm(norm1.transpose(0, 1), norm2)
        norm_sim_matrix = sim_matrix / (norm_matrix * self.temperature)
        sim_matrix_exp = torch.exp(norm_sim_matrix)
        if key_ids is None:
            key_ids = torch.arange(0, len(embeddings_v1))
        # find all positive pairs
        key_ids1, key_ids2 = torch.meshgrid(key_ids, key_ids, indexing='ij')
        pos_mask = key_ids1 == key_ids2

        # (1) get loss resulting from considering view 1 embeds to be queries and v2 to be keys
        # get denominator --> sum rows
        row_sum = torch.sum(sim_matrix_exp, dim=1)
        # normalize sim_matrix by row_sum denominator
        sim_matrix_row_divide = sim_matrix_exp / row_sum[:, None].repeat(1, sim_matrix_exp.shape[1])
        # take mean loss from all positive pairs
        view1_loss = -torch.mean(torch.log(sim_matrix_row_divide[pos_mask]))

        # (2) get loss resulting from considering view 2 embeds to be queries and v1 to be keys
        col_sum = torch.sum(sim_matrix_exp, dim=0)
        # normalize sim_matrix by col_sum denominator
        sim_matrix_col_divide = sim_matrix_exp / col_sum.repeat(sim_matrix_exp.shape[0], 1)
        # take mean loss from all positive pairs
        view2_loss = -torch.mean(torch.log(sim_matrix_col_divide[pos_mask]))

        loss = (view1_loss + view2_loss) / 2
        return loss
