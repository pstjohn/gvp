import torch

from torch_gvp.nn.layers import RBF


def test_rbf():
    input_ = 3 * torch.rand((10, 4), dtype=torch.float32)
    rbf_layer = RBF(dimension=12, init_max_distance=10)
    output = rbf_layer(input_)
    assert output.shape == (10, 4, 12)
    assert output.max() < 1.0
