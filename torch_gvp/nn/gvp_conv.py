from typing import List, Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from torch_gvp.nn import layers as gvp_layers
from torch_gvp.nn import vector
from torch_gvp.nn.gvp import GVP, GVPVectorGate
from torch_gvp.typing import ActivationFnArgs, VectorTuple, VectorTupleDim


def build_gvp_stack(
    n_layers: int,
    in_dims: VectorTupleDim,
    hid_dims: VectorTupleDim,
    out_dims: VectorTupleDim,
    activations: Optional[ActivationFnArgs],
    vector_gate: bool,
) -> List[GVP]:
    """Builds a stack of GVP layers with appropriate activation functions given the
    desired input and output dimensions

    Parameters
    ----------
    n_layers : int
        number of GVP layers per message update
    in_dims : VectorTupleDim
        Dimensions of the input node scalar and vector features
    out_dims : VectorTupleDim
        Dimensions of the output node scalar and vector features
    edge_dims : VectorTupleDim
        Dimensions of the edge's scalar and vector features
    activations : ActivationFnArgs
        Tuple of activation functions to use in intermediate layers
    vector_gate : bool
        Whether to use the GVPVectorGate class

    Returns
    -------
    List[GVP]
        A list of GVP or GVPVectorGate classes representing the DNN stack
    """

    gvp_class = GVPVectorGate if vector_gate else GVP

    module_list = []

    for i in range(n_layers):

        # No activation on the last layer
        if i == n_layers - 1:
            activations = (None, None)

        layer_in_dims = in_dims if i == 0 else hid_dims
        layer_out_dims = hid_dims if i != n_layers - 1 else out_dims

        module_list.append(
            gvp_class(
                layer_in_dims,
                layer_out_dims,
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
        activations: Optional[ActivationFnArgs] = None,
        vector_gate: bool = True,
    ):
        super().__init__(aggr=aggr)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.edge_dims = edge_dims

        if module_list is None:
            # The first layer needs to accept the concatenated message vector
            gvp_in_dims = (
                2 * in_dims[0] + edge_dims[0],
                2 * in_dims[1] + edge_dims[1],
            )
            module_list = build_gvp_stack(
                n_layers,
                gvp_in_dims,
                out_dims,
                out_dims,
                activations,
                vector_gate,
            )

        self.message_stack = gvp_layers.VectorSequential(*module_list)

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
        s, v = self.message_stack(s, v)

        return vector.merge(s, v)


class GVPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param gvp_conv_n_layers: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        node_dims: VectorTupleDim,
        edge_dims: VectorTupleDim,
        n_message: int = 3,
        n_feedforward: int = 2,
        drop_rate: float = 0.1,
        activations: Optional[ActivationFnArgs] = None,
        vector_gate: bool = True,
    ):

        super().__init__()
        self.conv = GVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_layers=n_message,
            aggr="mean",
            activations=activations,
            vector_gate=vector_gate,
        )

        self.vector_norm = nn.ModuleList(
            [gvp_layers.LayerNorm(node_dims) for _ in range(2)]
        )
        self.vector_dropout = nn.ModuleList(
            [gvp_layers.Dropout(drop_rate) for _ in range(2)]
        )

        ff_stack = build_gvp_stack(
            n_layers=n_feedforward,
            in_dims=node_dims,
            hid_dims=(4 * node_dims[0], 2 * node_dims[1]),
            out_dims=node_dims,
            activations=activations,
            vector_gate=vector_gate,
        )

        self.ff_stack = gvp_layers.VectorSequential(*ff_stack)

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> VectorTuple:

        # graph convolutional layer
        d_node_s, d_node_v = self.conv(node_s, node_v, edge_s, edge_v, edge_index)
        d_node_s, d_node_v = self.vector_dropout[0](d_node_s, d_node_v)
        node_s, node_v = vector.tuple_sum((node_s, node_v), (d_node_s, d_node_v))
        node_s, node_v = self.vector_norm[0](node_s, node_v)

        if len(self.ff_stack) == 0:
            return node_s, node_v

        # node-level update with FF GVP
        d_node_s, d_node_v = self.ff_stack(node_s, node_v)
        d_node_s, d_node_v = self.vector_dropout[1](d_node_s, d_node_v)
        node_s, node_v = vector.tuple_sum((node_s, node_v), (d_node_s, d_node_v))
        node_s, node_v = self.vector_norm[1](node_s, node_v)

        return node_s, node_v
