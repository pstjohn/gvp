from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from gvp.nn.vector import norm_no_nan
from gvp.typing import ActivationFnArgs, VectorTuple, VectorTupleDim


class GVP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims: VectorTupleDim,
        out_dims: VectorTupleDim,
        h_dim: Optional[int] = None,
        activations: ActivationFnArgs = (F.relu, torch.sigmoid),
        vector_gate: bool = False,
    ):
        super(GVP, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.vector_gate = vector_gate
        self.h_dim = h_dim or max(in_dims[1], out_dims[1])

        self.wh = nn.Linear(in_dims[1], self.h_dim, bias=False)
        self.ws = nn.Linear(self.h_dim + in_dims[0], out_dims[0])
        self.wv = nn.Linear(self.h_dim, out_dims[1], bias=False)
        if self.vector_gate:
            self.wsv = nn.Linear(*out_dims)

        self.scalar_act, self.vector_act = activations

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> VectorTuple:
        """Forward pass of the geometric vector perceptron layer

        Parameters
        ----------
        s : torch.Tensor
            Scalar features of size (n_batch, n_scalar_feat)
        v : torch.Tensor
            Vector features of size (n_batch, n_vector_feat, 3)

        Returns
        -------
        VectorTuple
            scalar and vector features after transformation
        """
        v = torch.transpose(v, -1, -2)
        vh = self.wh(v)
        vn = norm_no_nan(vh, axis=-2)
        s = self.ws(torch.cat([s, vn], -1))

        v = self.wv(vh)
        v = torch.transpose(v, -1, -2)

        # TODO: does it make more sense to do handle this split via subclassing?vi
        if self.vector_gate:
            if self.vector_act:
                gate = self.wsv(self.vector_act(s))
            else:
                gate = self.wsv(s)
            v = v * torch.sigmoid(gate).unsqueeze(-1)

        elif self.vector_act:
            v = v * self.vector_act(norm_no_nan(v, axis=-1, keepdims=True))

        if self.scalar_act:
            s = self.scalar_act(s)

        return s, v
