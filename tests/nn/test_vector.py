import pytest
import torch

from torch_gvp.nn.vector import merge, split
from torch_gvp.test.data import rand_vector_tuple


@pytest.mark.parametrize("dim", [(0, 0), (16, 8), (0, 16), (16, 0)])
def test_split_merge(dim):

    s, v = rand_vector_tuple(10, dim)
    s_out, v_out = split(merge(s, v), dim[1])
    assert torch.allclose(s, s_out)
    assert torch.allclose(v, v_out)
