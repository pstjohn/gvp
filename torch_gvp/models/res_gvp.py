from typing import Optional

import torch
from torch import nn

from torch_gvp.data.residue import AtomType, ResidueType
from torch_gvp.nn import layers as gvp_layers
from torch_gvp.nn.gvp import GVP
from torch_gvp.nn.gvp_conv import GVPConvLayer
from torch_gvp.nn.pool import vector_mean_pool
from torch_gvp.typing import ActivationFnArgs, VectorTupleDim


class ResidueGVP(nn.Module):
    def __init__(
        self,
        node_dims: VectorTupleDim = (128, 16),
        edge_dims: VectorTupleDim = (32, 1),
        n_atom_conv: int = 1,
        n_res_conv: int = 3,
        conv_n_message: int = 3,
        conv_n_feedforward: int = 2,
        drop_rate: float = 0.1,
        activations: Optional[ActivationFnArgs] = None,
        vector_gate: bool = True,
        init_max_distance: float = 10.0,
    ) -> None:
        super().__init__()

        self.n_atom_conv = n_atom_conv
        self.n_res_conv = n_res_conv

        self.conv_layers = nn.ModuleList(
            [
                GVPConvLayer(
                    node_dims=node_dims,
                    edge_dims=edge_dims,
                    n_message=conv_n_message,
                    n_feedforward=conv_n_feedforward,
                    drop_rate=drop_rate,
                    activations=activations,
                    vector_gate=vector_gate,
                )
                for _ in range(n_atom_conv + n_res_conv)
            ]
        )

        self.atom_embedding = nn.Embedding(
            len(AtomType) + 1, node_dims[0], padding_idx=0
        )
        self.res_embedding = nn.Embedding(
            len(ResidueType) + 1, node_dims[0], padding_idx=0
        )
        self.edge_rbf = gvp_layers.RBF(
            dimension=edge_dims[0], init_max_distance=init_max_distance, trainable=True
        )

        self.first_node_layer = gvp_layers.VectorSequential(
            GVP((node_dims[0], 2), node_dims, activations=(None, None)),
            gvp_layers.LayerNorm(node_dims),
        )

        self.first_edge_layer = gvp_layers.VectorSequential(
            GVP((edge_dims[0], 1), edge_dims, activations=(None, None)),
            gvp_layers.LayerNorm(edge_dims),
        )

        # Need 21 output classes to handle the unknown token as well
        self.masked_seq_head = GVP(node_dims, (21, 0), activations=(None, None))

    def forward(
        self,
        atom_type: torch.Tensor,
        residue_type: torch.Tensor,
        node_v: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
        edge_index: torch.Tensor,
        residue_index: torch.Tensor,
    ) -> torch.Tensor:

        node_s = self.atom_embedding(atom_type) + self.res_embedding(residue_type)
        edge_s = self.edge_rbf(edge_s)

        node_s, node_v = self.first_node_layer(node_s, node_v)
        edge_s, edge_v = self.first_edge_layer(edge_s, edge_v)

        for i in range(self.n_atom_conv):
            node_s, node_v = self.conv_layers[i](
                node_s, node_v, edge_s, edge_v, edge_index
            )

        node_s, node_v, edge_s, edge_v, edge_index = vector_mean_pool(
            node_s, node_v, edge_s, edge_v, edge_index, residue_index,
        )

        for i in range(self.n_res_conv):
            node_s, node_v = self.conv_layers[self.n_atom_conv + i](
                node_s, node_v, edge_s, edge_v, edge_index
            )

        out, _ = self.masked_seq_head(node_s, node_v)
        return out
