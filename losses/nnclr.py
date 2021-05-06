import torch
import torch.nn.functional as F


def nnclr_loss_func(nn, p, temp=0.1):
    nn = F.normalize(nn, dim=1)
    p = F.normalize(p, dim=1)

    # same as logits = nn @ p.T / temp
    logits = torch.einsum("if, jf -> ij", nn, p) / temp

    n = p.size(0)
    labels = torch.arange(n, device=p.device)

    loss = F.cross_entropy(logits, labels)
    return loss
