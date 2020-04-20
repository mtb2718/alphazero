import torch
from torch import nn
import torch.nn.functional as F


def masked_kl_div(input, target, mask):
    """Evaluate masked KL divergence between input activations and target distribution.

    Parameters:
        input (tensor) - NxD batch of D-dimensional activations (un-normalized log distribution).
        target (tensor) - NxD normalized target distribution.
        mask (tensor, torch.bool) - NxD mask of elements to include in calculation.

    Returns: Nx1 tensor of cross-entropy calculation results.
    """

    input = input.clone()
    input[~mask] = -float('inf')
    log_q = F.log_softmax(input, dim=1)
    log_q[~mask] = 0
    log_p = torch.log(target)
    log_p[~mask] = 0
    KLi = target * (log_p - log_q)
    KLi[target == 0] = 0
    KL = torch.sum(KLi, dim=1, keepdim=True)
    return KL


def masked_cross_entropy(input, target, mask):
    """Evaluate masked cross entry between input activations and target distribution.

    Parameters:
        input (tensor) - NxD batch of D-dimensional activations (un-normalized log distribution).
        target (tensor) - NxD normalized target distribution.
        mask (tensor, torch.bool) - NxD mask of elements to include in calculation.

    Returns: Nx1 tensor of cross-entropy calculation results.
    """

    input = input.clone()
    input[~mask] = -float('inf')
    log_q = F.log_softmax(input, dim=1)
    log_q[~mask] = 0
    H = -torch.sum(target * log_q, dim=1, keepdim=True)
    return H


class AlphaZeroLoss(nn.Module):

    def forward(self, p, z, p_hat, v_hat, p_valid):
        B = p.shape[0]

        # Calculate loss
        # Value
        value_loss = ((z - v_hat) ** 2).mean()

        # Policy
        assert torch.all(torch.abs(p.sum(dim=1) - 1.) < 1e-6), \
            f'Target policy must be a valid distribution: {p}'
        policy_loss = masked_cross_entropy(p_hat, p, p_valid).mean()
        return policy_loss, value_loss
