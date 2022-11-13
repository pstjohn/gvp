import pytest
import torch
from scipy.spatial.transform import Rotation

from gvp.nn.gvp import GVP
from gvp.nn.gvp_conv import GVPConv
from gvp.test.data import rand_vector_tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def rotation():
    return torch.as_tensor(
        Rotation.random().as_matrix(), dtype=torch.float32, device=device
    )


@pytest.mark.parametrize("v_in,v_out", [(0, 0), (16, 8), (0, 16), (16, 0)])
@pytest.mark.parametrize("vector_gate", [True, False])
def test_gvp(rotation, v_in, v_out, vector_gate):
    node_dim_in = (32, v_in)
    node_dim_out = (24, v_out)
    num_nodes = 10
    s, v = rand_vector_tuple(num_nodes, node_dim_in)

    model = GVP(node_dim_in, node_dim_out, vector_gate=vector_gate).to(device).eval()

    with torch.no_grad():
        s_out, v_out = model(s, v)

    assert s_out.shape == (num_nodes, node_dim_out[0])
    assert v_out.shape == (num_nodes, node_dim_out[1], 3)

    # These should be all zeros if the input v was empty
    if node_dim_in[1] == 0:
        assert torch.allclose(v_out, torch.zeros_like(v_out))

    # If we don't have an output, then this vector should be empty
    if node_dim_out[1] == 0:
        assert v_out.nelement() == 0

    # Check equivariance
    with torch.no_grad():
        v_rot = v @ rotation
        out_v_rot = v_out @ rotation
        out_s_prime, out_v_prime = model(s, v_rot)
        assert torch.allclose(s_out, out_s_prime, atol=1e-5, rtol=1e-4)
        assert torch.allclose(out_v_rot, out_v_prime, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("v_in,v_out", [(0, 0), (16, 8), (0, 16), (16, 0)])
@pytest.mark.parametrize("vector_gate", [True, False])
def test_gvp_conv(rotation, v_in, v_out, vector_gate):
    node_dim_in = (32, v_in)
    node_dim_out = (24, v_out)
    edge_dim = (32, 1)
    num_nodes = 10
    num_edges = 25
    n_layers = 2

    node_s, node_v = rand_vector_tuple(num_nodes, node_dim_in)
    edge_s, edge_v = rand_vector_tuple(num_edges, edge_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    model = (
        GVPConv(node_dim_in, node_dim_out, edge_dim, n_layers, vector_gate=vector_gate)
        .to(device)
        .eval()
    )

    with torch.no_grad():
        s_out, v_out = model(node_s, node_v, (edge_s, edge_v), edge_index)

    assert s_out.shape == (num_nodes, node_dim_out[0])
    assert v_out.shape == (num_nodes, node_dim_out[1], 3)

    # Check equivariance
    with torch.no_grad():
        node_v_rot = node_v @ rotation
        edge_v_rot = edge_v @ rotation

        s_out_prime, v_out_prime = model(
            node_s, node_v_rot, (edge_s, edge_v_rot), edge_index
        )

        assert torch.allclose(s_out, s_out_prime, atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_out @ rotation, v_out_prime, atol=1e-5, rtol=1e-4)
