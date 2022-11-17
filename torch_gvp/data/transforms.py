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


def _normalize(tensor: torch.Tensor, dim=-1) -> torch.Tensor:
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _orientations(X: torch.Tensor, discont: torch.Tensor) -> torch.Tensor:
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])

    # For chain discontinuities, zero these input directions
    forward[discont - 1] = 0
    backward[discont] = 0

    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


class NodeOrientation(BaseTransform):
    def __call__(self, data: Data) -> Data:

        dist = _orientations(data.pos, data.residue_discont)
        data["node_v"] = dist
        del data["residue_discont"]

        return data


class ResidueMask(BaseTransform):
    def __init__(self, mask_prob: float = 0.4, random_token_prob: float = 0.15) -> None:
        """Randomly mask protein residues, recording the true residue types and the
        boolean node and residue mask.

        true_residue_type only records the masked values. Therefore the model should
        predict `all_predicted_residue_type[residue_mask] = true_residue_type`

        Parameters
        ----------
        mask_prob : float, optional
            The faction of input residues to mask, by default 0.4
        random_token_prob: float, optional
            The chance of using a random AA token instead of the mask token
        """
        self.mask_prob = mask_prob
        self.random_token_prob = random_token_prob

    def __call__(self, data: Data) -> Data:
        assert data.num_nodes is not None, "Ensure num_nodes for typing"
        data["residue_mask"] = torch.rand(data.num_nodes // 3) < self.mask_prob
        data["node_mask"] = data["residue_mask"].repeat_interleave(3)
        data["true_residue_type"] = data.residue_type[::3][data["residue_mask"]]

        # Here, we generate a random masking vector where we mask residues either with
        # the mask token p=(1 - random_token_prob), or a randomly chosen AA in 1-20
        num_masked_residues = data["residue_mask"].sum().item()
        random_token = torch.randint(
            low=1, high=21, size=(num_masked_residues,), dtype=torch.int32
        )
        mask_token = ResidueType.MASK.value * torch.ones_like(random_token)
        reside_mask_tokens = torch.where(
            torch.rand((num_masked_residues,)) < self.random_token_prob,
            random_token,
            mask_token,
        )

        data["residue_type"][data["node_mask"]] = reside_mask_tokens.repeat_interleave(
            3
        )
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
