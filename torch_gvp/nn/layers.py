from typing import Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter

from torch_gvp.nn.vector import norm_no_nan
from torch_gvp.typing import VectorTuple


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.parameter.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> VectorTuple:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims: Tuple[int, int]):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> VectorTuple:
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        vn = norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class RBF(nn.Module):
    def __init__(
        self,
        dimension: int = 64,
        init_max_distance: float = 10.0,
        trainable: bool = True,
    ) -> None:
        """Computes a trainable Gaussian radial basis function expansion of the inputs.
        For a given input, returns an output of shape [..., dimension] expanded on the
        last dimension. Adapted from github.com/atomistic-machine-learning/schnetpack.

        Original copyright:

        Copyright (c) 2018 Kristof Schütt, Michael Gastegger, Pan Kessel, Kim Nicoli

        All other contributions:
        Copyright (c) 2018, the respective contributors.
        All rights reserved.

        Parameters
        ----------
        dimension : int, optional
            Size of the output embedding, by default 64
        init_max_distance : float, optional
            Maximum distance of the RBF embedding buckets, by default 10.0
        trainable : bool, optional
            Whether parameters should be fit during backprop, by default True
        """
        super().__init__()

        offsets = torch.linspace(0, init_max_distance, dimension, dtype=torch.float32)
        widths = torch.Tensor(init_max_distance / dimension * torch.ones_like(offsets))

        if trainable:
            self.offsets = Parameter(offsets)
            self.widths = Parameter(widths)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("widths", widths)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        coeff = -0.5 / torch.pow(self.widths, 2)
        diff = inputs[..., None] - self.offsets
        y = torch.exp(coeff * torch.pow(diff, 2))
        return y


class VectorSequential(nn.Sequential):
    def forward(self, s: torch.Tensor, v: torch.Tensor) -> VectorTuple:
        for module in self:
            s, v = module(s, v)
        return s, v
