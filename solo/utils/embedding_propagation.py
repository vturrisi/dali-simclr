# Code from https://github.com/ServiceNow/embedding-propagation

import torch
import torch.nn.functional as F
import numpy as np


def euclidean_across_channels(x1, x2):
    """
    Given two input X1, X2 of shape (c, h, w), flattens the last two dimensions and
    calculates the euclidean distance between pairs of channels and returns the average 
    """
    c, h, w = x1.size()
    x1 = x1.view(c, -1)
    x2 = x2.view(c, -1)
    return ((x1 - x2) ** 2).sum(dim=-1).mean(dim=-1)

def get_channel_euclidean_similarity_matrix(x, rbf_scale, scaling_factor=True):
    b, c, _, _ = x.size()
    sq_dist = torch.zeros(b, b, device=x.device)
    for i in range(b):
        for j in range(b):
            dist = euclidean_across_channels(x[i], x[j])
            sq_dist[i, j] = dist
            sq_dist[j, i] = dist
    if scaling_factor:
        sq_dist  = sq_dist / np.sqrt(c)
    mask = sq_dist != 0
    sq_dist = sq_dist / sq_dist[mask].std()
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
    weights = weights * (~mask).float()
    return weights

def get_cosine_similarity_matrix(x):
    """
    Given input X of shape (b, c, h, w) computes a similarity matrix based on cosine similarity.
    For two inputs x1 and x2, we compute like this:
    compute the similarity between each channel in x1 and x2
    average the corresponding similarities.

    We do this for all pairwise samples in X.
    """
    b, c, _, _ = x.size()
    x = x.view(b, c, -1)
    x = F.normalize(x, dim=2)
    similarity_matrix = (x.view(b, 1, c, -1) * x.view(1, b, c, -1)).sum(dim=-1).mean(dim=-1)
    mask = torch.eye(similarity_matrix.size(1), dtype=torch.bool, device=similarity_matrix.device)
    return (1 + similarity_matrix) * (~mask).float()


def get_similarity_matrix(x, rbf_scale, scaling_factor=True, fixed_gamma=None):
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c)) ** 2).sum(-1)
    if scaling_factor:
        sq_dist  = sq_dist / np.sqrt(c)
    mask = sq_dist != 0
    gamma = sq_dist[mask].std()
    if fixed_gamma:
        sq_dist = sq_dist / fixed_gamma
    else:
        sq_dist = sq_dist / gamma
    weights = torch.exp(-sq_dist * rbf_scale)
    mask = torch.eye(weights.size(1), dtype=torch.bool, device=weights.device)
    weights = weights * (~mask).float()
    return weights, gamma

def get_distance_matrix(x):
    b, c = x.size()
    sq_dist = ((x.view(b, 1, c) - x.view(1, b, c)) ** 2).sum(-1)
    return sq_dist

def get_laplacian(weights, normalized=True):
    if normalized:
        # According to the paper, normalized laplacian might work better
        isqrt_diag = 1.0 / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
        # checknan(laplacian=isqrt_diag)
        S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
        return torch.eye(weights.shape[0], device=weights.device) - S
    else:
        return torch.diag(weights.sum(dim=-1)) - weights


def embedding_propagation(x, alpha, rbf_scale, norm_prop, propagator=None):
    if propagator is None:
        weights = get_similarity_matrix(x, rbf_scale)
        propagator = global_consistency(weights, alpha=alpha, norm_prop=norm_prop)
    return torch.mm(propagator, x)


def global_consistency(weights, alpha=1, norm_prop=False):
    """Implements D. Zhou et al. "Learning with local and global consistency". (Same as in TPN paper but without bug)

    Args:
        weights: Tensor of shape (n, n). Expected to be exp( -d^2/s^2 ), where d is the euclidean distance and
            s the scale parameter.
        labels: Tensor of shape (n, n_classes)
        alpha: Scaler, acts as a smoothing factor
    Returns:
        Tensor of shape (n, n_classes) representing the logits of each classes
    """
    n = weights.shape[1]
    identity = torch.eye(n, dtype=weights.dtype, device=weights.device)
    isqrt_diag = 1.0 / torch.sqrt(1e-4 + torch.sum(weights, dim=-1))
    # checknan(laplacian=isqrt_diag)
    S = weights * isqrt_diag[None, :] * isqrt_diag[:, None]
    # checknan(normalizedlaplacian=S)
    propagator = identity - alpha * S
    propagator = torch.inverse(propagator[None, ...])[0]
    # checknan(propagator=propagator)
    if norm_prop:
        propagator = F.normalize(propagator, p=1, dim=-1)
    return propagator
