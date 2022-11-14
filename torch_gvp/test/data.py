from typing import Union

import torch
from torch_geometric.data import Data

from torch_gvp.typing import VectorTuple, VectorTupleDim


def rand_vector_tuple(
    n: int, dims: VectorTupleDim, device: Union[torch.device, str] = "cpu"
) -> VectorTuple:
    """
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    """
    return (
        torch.randn(n, dims[0], device=device),
        torch.randn(n, dims[1], 3, device=device),
    )


def random_data(
    n_nodes: int,
    n_edges: int,
    node_dim: VectorTupleDim,
    edge_dim: VectorTupleDim,
    device: Union[torch.device, str] = "cpu",
) -> Data:

    return Data(
        x=rand_vector_tuple(n_nodes, node_dim),  # type: ignore
        edge_attr=rand_vector_tuple(n_edges, edge_dim),  # type: ignore
        edge_index=torch.randint(0, n_nodes, (2, n_edges), device=device),
    )
