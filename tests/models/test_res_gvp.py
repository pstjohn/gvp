import pytest
import torch

from torch_gvp.data.transforms import create_gvp_transformer_stack
from torch_gvp.models.res_gvp import ResidueGVP


@pytest.fixture
def model(device):
    return (
        ResidueGVP(
            node_dims=(32, 8),
            edge_dims=(8, 4),
            n_atom_conv=1,
            n_res_conv=1,
            conv_n_message=1,
            conv_n_feedforward=1,
        )
        .to(device)
        .eval()
    )


def test_residue_gvp(rcsb_loader, rotation, model, device):

    item = next(iter(rcsb_loader))
    item.to(device)

    with torch.no_grad():
        s_out = model(
            item.atom_type,
            item.residue_type,
            item.node_v,
            item.edge_s,
            item.edge_v,
            item.edge_index,
            item.residue_index,
        )

        s_out_prime = model(
            item.atom_type,
            item.residue_type,
            item.node_v @ rotation,
            item.edge_s,
            item.edge_v @ rotation,
            item.edge_index,
            item.residue_index,
        )

    assert s_out.shape[-1] == 20
    assert torch.allclose(s_out, s_out_prime, atol=1e-5, rtol=1e-4)
    assert s_out[item.residue_mask].shape[0] == item.true_residue_type.shape[0]


def test_multichain_prot(prot_data, model, device):

    xformer = create_gvp_transformer_stack(jitter=0.02, residue_mask_prob=0.35)
    item = xformer(prot_data)
    item.to(device)

    with torch.no_grad():
        s_out = model(
            item.atom_type,
            item.residue_type,
            item.node_v,
            item.edge_s,
            item.edge_v,
            item.edge_index,
            item.residue_index,
        )

    assert s_out[item.residue_mask].shape[0] == item.true_residue_type.shape[0]
