import torch

from torch_gvp.models.res_gvp import ResidueGVP


def test_residue_gvp(rcsb_loader, rotation, device):
    model = (
        ResidueGVP(
            node_dims=(32, 8),
            edge_dims=(8, 4),
            n_conv=2,
            conv_n_message=1,
            conv_n_feedforward=1,
        )
        .to(device)
        .eval()
    )

    item = next(iter(rcsb_loader))

    with torch.no_grad():
        s_out = model(
            item.atom_type,
            item.residue_type,
            item.node_v,
            item.edge_s,
            item.edge_v,
            item.edge_index,
        )

        s_out_prime = model(
            item.atom_type,
            item.residue_type,
            item.node_v @ rotation,
            item.edge_s,
            item.edge_v @ rotation,
            item.edge_index,
        )

    assert s_out.shape[-1] == 20
    assert torch.allclose(s_out, s_out_prime, atol=1e-5, rtol=1e-4)
