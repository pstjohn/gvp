from typing import List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from gvp.nn import vector
from gvp.nn.gvp import GVP, GVPVectorGate
from gvp.typing import ActivationFnArgs, VectorTuple, VectorTupleDim


def build_gvp_module_list(
    n_layers: int,
    in_dims: VectorTupleDim,
    out_dims: VectorTupleDim,
    edge_dims: VectorTupleDim,
    activations: ActivationFnArgs,
    vector_gate: bool,
) -> List[GVP]:

    si, vi = in_dims
    se, ve = edge_dims
    gvp_in_dims = (2 * si + se, 2 * vi + ve)
    gvp_class = GVPVectorGate if vector_gate else GVP

    module_list = []

    for i in range(n_layers):

        # No activation on the last layer
        if i == n_layers - 1:
            activations = (None, None)

        # The first layer needs to accept the concatenated message vectors
        layer_in_dims = gvp_in_dims if i == 0 else out_dims

        module_list.append(
            gvp_class(
                layer_in_dims,
                out_dims,
                activations=activations,
            )
        )

    return module_list


class GVPConv(MessagePassing):
    """
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims: VectorTupleDim,
        out_dims: VectorTupleDim,
        edge_dims: VectorTupleDim,
        n_layers: int = 3,
        module_list: Optional[List[GVP]] = None,
        aggr: str = "mean",
        activations: ActivationFnArgs = (F.relu, torch.sigmoid),
        vector_gate: bool = False,
    ):
        super(GVPConv, self).__init__(aggr=aggr)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.edge_dims = edge_dims

        if module_list is None:
            module_list = build_gvp_module_list(
                n_layers, in_dims, out_dims, edge_dims, activations, vector_gate
            )

        self.module_list = module_list

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> VectorTuple:
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        message = self.propagate(
            edge_index,
            s=node_s,
            v=node_v.reshape(node_v.shape[0], 3 * node_v.shape[1]),
            edge_s=edge_s,
            edge_v=edge_v,
        )
        return vector.split(message, self.out_dims[1])

    def message(
        self,
        s_i: torch.Tensor,
        v_i: torch.Tensor,
        s_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        s, v = vector.tuple_cat((s_j, v_j), (edge_s, edge_v), (s_i, v_i))

        for layer in self.module_list:
            s, v = layer(s, v)

        return vector.merge(s, v)
