import torch

from torch_gvp.nn.pool import vector_mean_pool
from torch_gvp.test.data import rand_vector_tuple


def test_vector_mean_pool(rotation, device):

    num_nodes = 12
    num_edges = 25

    node_dim = (10, 5)
    edge_dim = (4, 7)

    node_s, node_v = rand_vector_tuple(num_nodes, node_dim, device)
    edge_s, edge_v = rand_vector_tuple(num_edges, edge_dim, device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    cluster = torch.arange(4).repeat_interleave(3)

    with torch.no_grad():
        (
            node_s_pool,
            node_v_pool,
            edge_s_pool,
            edge_v_pool,
            edge_index_pool,
        ) = vector_mean_pool(
            node_s,
            node_v,
            edge_s,
            edge_v,
            edge_index,
            cluster,
        )

        (
            node_s_pool_prime,
            node_v_pool_prime,
            edge_s_pool_prime,
            edge_v_pool_prime,
            edge_index_pool_prime,
        ) = vector_mean_pool(
            node_s,
            node_v @ rotation,
            edge_s,
            edge_v @ rotation,
            edge_index,
            cluster,
        )

    assert node_s_pool.shape[0] == 4
    assert torch.allclose(node_s_pool, node_s_pool_prime)
    assert torch.allclose(node_v_pool @ rotation, node_v_pool_prime)
    assert torch.allclose(edge_s_pool, edge_s_pool_prime)
    assert torch.allclose(edge_v_pool @ rotation, edge_v_pool_prime)
    assert torch.allclose(edge_index_pool, edge_index_pool_prime)
