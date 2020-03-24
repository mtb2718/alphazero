import torch
from torch import nn
import torch.nn.functional as F


def masked_cross_entropy(input, target, mask):
    """Evaluate masked cross entry between input activations and target distribution.

    Parameters:
        input (tensor) - NxD batch of D-dimensional activations (un-normalized distribution).
        target (tensor) - NxD normalized target distribution.
        mask (tensor, torch.bool) - NxD mask of elements to include in calculation

    Returns: Nx1 tensor of cross-entropy calculation results.
    """

    # TODO: properly implement a masked cross-entropy loss
    input = input.clone()
    input[~mask] = -999
    log_q = F.log_softmax(input, dim=1)
    H = -torch.sum(target * log_q, dim=1, keepdim=True)
    return H


class AlphaZeroLoss(nn.Module):

    def forward(self, p, z, p_hat, v_hat, p_valid):
        B = p.shape[0]

        # Calculate loss
        # Value estimate
        value_loss = ((z - v_hat) ** 2).mean()

        # Prior policy
        assert torch.all(torch.abs(p.sum(dim=1) - 1.) < 1e-6), \
            f'Target prior must be a valid distribution. Got: {p}'
        prior_loss = masked_cross_entropy(p_hat, p, p_valid).mean()
        return prior_loss, value_loss
