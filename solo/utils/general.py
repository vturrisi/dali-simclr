import torch
import torch.nn as nn


def filter_inf_n_nan(tensor: torch.Tensor) -> torch.Tensor:
    """Filters out inf and nans from any tensor.
    This is usefull when there are instability issues,
    which cause a small number of values to go bad.

    Args:
        tensor (torch.Tensor): tensor to remove nans and infs from.

    Returns:
        torch.Tensor: filtered view of the tensor without nans or infs.
    """
    if len(tensor.size()) == 1:
        tensor = tensor[~tensor.isnan()]
        tensor = tensor[~tensor.isinf()]
    elif len(tensor.size()) == 2:
        tensor = tensor[~torch.any(tensor.isnan(), dim=1)]
        tensor = tensor[~torch.any(tensor.isinf(), dim=1)]

    return tensor


class FilterInfNNan(nn.Module):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.module(x)
        out = filter_inf_n_nan(out)
        return out

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "module":
                raise AttributeError()
            return getattr(self.module, name)
