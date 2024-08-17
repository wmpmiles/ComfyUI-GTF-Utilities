import torch
from typing import Iterable


def round_to_mult_of(number: int, mult_of: int) -> int:
    aligned = ((number + mult_of - 1) // mult_of) * mult_of
    return aligned


def gtf_min_max_f(
    is_max: bool,
) -> torch.Tensor:
    f = torch.amax if is_max else torch.amin

    def f(tensor: torch.Tensor, dims: Iterable[int]) -> torch.Tensor:
        curr = tensor
        for dim in dims:
            curr = f(curr, dim, True)
        return curr

    return f


gtf_min = gtf_min_max_f(False)
gtf_max = gtf_min_max_f(True)


def slice_dim(
    dim: int,
    start: int | None,
    stop: int | None,
    step: int | None = 1
) -> list[slice]:
    slices = [slice(None) for _ in range(dim)] + [slice(start, stop, step)]
    return slices


def invert(tensor: torch.Tensor) -> torch.Tensor:
    inverted = (1.0 - tensor)
    return inverted


def pad_tensor_reflect(
    tensor: torch.Tensor,
    dim: int,
    pad_start: int,
    pad_end: int
) -> torch.Tensor:
    length = tensor.shape[dim]
    start = tensor[slice_dim(dim, 1, pad_start+1)].flip(dim)
    end = tensor[slice_dim(dim, length-(pad_end+1), -1)].flip(dim)
    padded = torch.cat((start, tensor, end), dim)
    return padded
