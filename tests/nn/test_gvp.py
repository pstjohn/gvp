import pytest
import torch

from torch_gvp.nn.gvp import GVP, GVPVectorGate
from torch_gvp.nn.gvp_conv import GVPConv, GVPConvLayer
from torch_gvp.test.data import rand_vector_tuple


@pytest.fixture(name="v_in", params=(0, 16))
def _v_in(request):
    return request.param


@pytest.fixture(name="v_out", params=(0, 8))
def _v_out(request):
    return request.param


@pytest.fixture(name="vector_gate", params=(True, False))
def _vector_gate(request):
    return request.param


def test_gvp(rotation, v_in, v_out, vector_gate, device):
    node_dim_in = (32, v_in)
    node_dim_out = (24, v_out)
    num_nodes = 10
    s, v = rand_vector_tuple(num_nodes, node_dim_in)

    gvp_class = GVPVectorGate if vector_gate else GVP
    model = gvp_class(node_dim_in, node_dim_out).to(device).eval()

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


def test_gvp_conv(rotation, v_in, v_out, vector_gate, device):
    node_dim_in = (32, v_in)
    node_dim_out = (24, v_out)
    edge_dim = (32, 1)
    num_nodes = 10
    num_edges = 25
    n_layers = 2

    node_s, node_v = rand_vector_tuple(num_nodes, node_dim_in, device)
    edge_s, edge_v = rand_vector_tuple(num_edges, edge_dim, device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    model = (
        GVPConv(node_dim_in, node_dim_out, edge_dim, n_layers, vector_gate=vector_gate)
        .to(device)
        .eval()
    )

    with torch.no_grad():
        s_out, v_out = model(node_s, node_v, edge_s, edge_v, edge_index)

    assert s_out.shape == (num_nodes, node_dim_out[0])
    assert v_out.shape == (num_nodes, node_dim_out[1], 3)

    # Check equivariance
    with torch.no_grad():
        node_v_rot = node_v @ rotation
        edge_v_rot = edge_v @ rotation

        s_out_prime, v_out_prime = model(
            node_s, node_v_rot, edge_s, edge_v_rot, edge_index
        )

        assert torch.allclose(s_out, s_out_prime, atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_out @ rotation, v_out_prime, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("n_feedfoward", [0, 1])
def test_gvp_conv_layer(rotation, v_in, v_out, n_feedfoward, device):
    node_dim = (32, v_in)
    edge_dim = (32, v_out)
    num_nodes = 10
    num_edges = 25

    node_s, node_v = rand_vector_tuple(num_nodes, node_dim, device)
    edge_s, edge_v = rand_vector_tuple(num_edges, edge_dim, device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    model = (
        GVPConvLayer(node_dim, edge_dim, 1, n_feedfoward, vector_gate=True)
        .to(device)
        .eval()
    )

    with torch.no_grad():
        s_out, v_out = model(node_s, node_v, edge_s, edge_v, edge_index)

    assert s_out.shape == (num_nodes, node_dim[0])
    assert v_out.shape == (num_nodes, node_dim[1], 3)

    # Check equivariance
    with torch.no_grad():
        node_v_rot = node_v @ rotation
        edge_v_rot = edge_v @ rotation

        s_out_prime, v_out_prime = model(
            node_s, node_v_rot, edge_s, edge_v_rot, edge_index
        )

        assert torch.allclose(s_out, s_out_prime, atol=1e-5, rtol=1e-4)
        assert torch.allclose(v_out @ rotation, v_out_prime, atol=1e-5, rtol=1e-4)
