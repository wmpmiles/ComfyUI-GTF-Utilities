import torch
from typing import Iterable


def round_to_mult_of(number: int, mult_of: int) -> int:
    aligned = ((number + mult_of - 1) // mult_of) * mult_of
    return aligned


def gtf_min_max(tensor: torch.Tensor, dims: Iterable[int], max: bool = False) -> torch.Tensor:
    f = torch.max if max else torch.min
    curr = tensor
    for dim in dims:
        curr = f(curr, dim, True)[0]
    return curr


def slice_dim(
    dim: int, 
    start: int | None, 
    stop: int | None, 
    step: int | None = 1
) -> list[slice]:
    slices = [slice(None) for _ in range(dim)] + [slice(start, stop, step)]
    return slices


def tensor_invert(tensor: torch.Tensor) -> torch.Tensor:
    inverted = (1.0 - tensor)
    return inverted

