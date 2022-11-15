"""Orientation code adapted from drorlab/gvp-pytorch,
https://github.com/drorlab/gvp-pytorch/blob/main/gvp/data.py
"""

from typing import Optional

import torch
import torch.nn.functional as F
import torch_geometric.transforms
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from torch_gvp.data.residue import ResidueType


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


class ResidueMask(BaseTransform):
    def __init__(self, mask_prob: float = 0.4) -> None:
        """Randomly mask protein residues, recording the true residue types and the
        boolean node and residue mask.

        true_residue_type only records the masked values. Therefore the model should
        predict `all_predicted_residue_type[residue_mask] = true_residue_type`

        Parameters
        ----------
        mask_prob : float, optional
            The faction of input residues to mask, by default 0.4
        """
        self.mask_prob = mask_prob

    def __call__(self, data: Data) -> Data:
        assert data.num_nodes is not None, "Ensure num_nodes for typing"
        data["residue_mask"] = torch.rand(data.num_nodes // 3) < self.mask_prob
        data["node_mask"] = data["residue_mask"].repeat_interleave(3)
        data["true_residue_type"] = data.residue_type[::3][data["residue_mask"]]
        data.residue_type[data["node_mask"]] = ResidueType.MASK.value
        return data


class EdgeSplit(BaseTransform):
    """Splits the edge attributes into vector and scalar features.

    Assumes the first 3 entries are the vectorized distances, while the final is a
    scalar distance.

    Appends to data:
        edge_s: (batch_size,) float tensor
            Array of edge lengths (in Angstroms)
        edge_v: (batch_size, 1, 3)
            A vector representing the cartesian difference between the two points
    """

    def __call__(self, data: Data) -> Data:
        assert data.edge_attr.shape[-1] == 4
        data["edge_s"] = data.edge_attr[:, 3]
        data["edge_v"] = data.edge_attr[:, :3].unsqueeze(-2)
        del data["edge_attr"]
        return data


def create_gvp_transformer_stack(
    radius: float = 10.0,
    max_num_neighbors: int = 32,
    jitter: Optional[float] = None,
    residue_mask_prob: Optional[float] = None,
):

    stack = []
    if jitter:
        stack.append(torch_geometric.transforms.RandomJitter(jitter))

    stack += [
        torch_geometric.transforms.RadiusGraph(
            r=radius, loop=False, max_num_neighbors=max_num_neighbors
        ),
        torch_geometric.transforms.Cartesian(),
        torch_geometric.transforms.Distance(norm=False),
        NodeOrientation(),
        EdgeSplit(),
    ]

    if residue_mask_prob is not None:
        stack.append(ResidueMask(residue_mask_prob))

    return torch_geometric.transforms.Compose(stack)
