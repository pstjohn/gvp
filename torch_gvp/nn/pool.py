from typing import Tuple

import torch
from torch_geometric.nn.pool.avg_pool import _avg_pool_x
from torch_geometric.nn.pool.pool import pool_edge

from torch_gvp.nn import vector


def vector_mean_pool(
    node_s: torch.Tensor,
    node_v: torch.Tensor,
    edge_s: torch.Tensor,
    edge_v: torch.Tensor,
    edge_index: torch.Tensor,
    cluster: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pools node and edge representations and edge connections according to a cluster
    index, typically mapping atoms to individual residues.

    Parameters
    ----------
    node_s : torch.Tensor
    node_v : torch.Tensor
    edge_s : torch.Tensor
    edge_v : torch.Tensor
    edge_index : torch.Tensor
    cluster : torch.Tensor
        An integer mapping of node and edge features to new residue-level clusters

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        pooled node, edge, and edge index tensors
    """

    node = vector.merge(node_s, node_v)
    edge_attr = vector.merge(edge_s, edge_v)

    pooled_node = _avg_pool_x(cluster, node)
    pooled_edge_index, pooled_edge_attr = pool_edge(cluster, edge_index, edge_attr)

    return (
        *vector.split(pooled_node, node_v.size(1)),
        *vector.split(pooled_edge_attr, edge_v.size(1)),  # type: ignore
        pooled_edge_index,
    )
