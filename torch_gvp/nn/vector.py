import torch

from torch_gvp.typing import VectorTuple


def tuple_sum(*args: VectorTuple) -> VectorTuple:
    """
    Sums any number of tuples (s, V) elementwise.
    """
    return tuple(map(sum, zip(*args)))  # type: ignore


def tuple_cat(*args: VectorTuple, dim: int = -1):
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x: VectorTuple, idx: int) -> VectorTuple:
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def split(x: torch.Tensor, nv: int) -> VectorTuple:
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    scalar_dims = x.shape[-1] - 3 * nv
    v = torch.reshape(x[..., scalar_dims:], x.shape[:-1] + (nv, 3))
    s = x[..., :scalar_dims]
    return s, v


def merge(s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


def norm_no_nan(
    x: torch.Tensor,
    axis: int = -1,
    keepdims: bool = False,
    eps: float = 1e-8,
    sqrt: bool = True,
) -> torch.Tensor:
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out
