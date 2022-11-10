"""Code adapted from drorlab/gvp-pytorch,
https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
"""

import torch
import torch.nn.functional as F
import torch_geometric.transforms
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


class NodeOrientation(BaseTransform):
    def __init__(self, norm: bool = True) -> None:
        """Calculates vectors for the input and output edges of each node as node vector
        features, assuming that pos[n] is linked to pos[n+1] (i.e., through a protein
        backbone structure)

        Parameters
        ----------
        norm : bool, optional
            Whether to return a vector with unit euclidian norm, by default True
        cat : bool, optional
            If set to :obj:`False`, all existing edge attributes will be replaced.
            (default: :obj:`True`)
        """
        self.norm = norm

    def __call__(self, data: Data) -> Data:

        dist = _orientations(data.pos)
        if self.norm:
            dist = _normalize(dist)

        data["node_v"] = dist

        return data


class GVPTransfomer(BaseTransform):
    """Splits the edge attributes into vector and scalar features.

    Assumes the first 3 entries are the vectorized distances, while the final is a
    scalar distance.
    """

    def __call__(self, data: Data) -> Data:

        assert data.edge_attr.shape[-1] == 4
        edge_s = data.edge_attr[:, 3].unsqueeze(-1)
        edge_v = data.edge_attr[:, :3].unsqueeze(-2)
        del data["edge_attr"]
        data["edge_s"] = edge_s
        data["edge_v"] = edge_v

        return data


def create_gvp_transformer_stack(radius: float = 10.0, max_num_neighbors: int = 32):

    return torch_geometric.transforms.Compose(
        [
            torch_geometric.transforms.Center(),
            torch_geometric.transforms.RadiusGraph(
                r=radius, loop=False, max_num_neighbors=max_num_neighbors
            ),
            torch_geometric.transforms.Cartesian(),
            torch_geometric.transforms.Distance(norm=False),
            NodeOrientation(),
            GVPTransfomer(),
        ]
    )
