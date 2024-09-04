import torch
from typing import Iterable


def round_up_to_mult_of(number: int, mult_of: int) -> int:
    aligned = ((number + mult_of - 1) // mult_of) * mult_of
    return aligned


def gtf_min(tensor: torch.Tensor, dims: Iterable[int]) -> torch.Tensor:
    curr = tensor
    for dim in dims:
        curr = torch.amin(curr, dim, True)
    return curr


def gtf_max(tensor: torch.Tensor, dims: Iterable[int]) -> torch.Tensor:
    curr = tensor
    for dim in dims:
        curr = torch.amax(curr, dim, True)
    return curr


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


def pad_reflect_radius(tensor: torch.Tensor, dims: tuple[int], radius: int) -> torch.Tensor:
    padded = tensor
    for dim in dims:
        padded = pad_tensor_reflect(padded, dim, radius, radius)
    return padded


def unfold(gtf: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    unfolded = gtf.unfold(3 , kw, 1).unfold(2, kh, 1)
    return unfolded


def outer_sum(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    ret = lhs.unsqueeze(1) + rhs.unsqueeze(0)
    return ret


def sum_normalize(
    tensor: torch.Tensor, dims: Iterable[int]
) -> torch.Tensor:
    dims = sorted(dims)
    denominator = tensor
    for dim in reversed(dims):
        denominator = denominator.sum(dim)
    for dim in dims:
        denominator = denominator.unsqueeze(dim)
    normalized = tensor / denominator
    return normalized


def range_normalize(
    tensor: torch.Tensor, dims: Iterable[int]
) -> torch.Tensor:
    cur_min = gtf_min(tensor, dims)
    minned = tensor - cur_min
    cur_max = gtf_max(minned, dims)
    maxxed = minned / cur_max
    clamped = maxxed.clamp(0, 1)
    return clamped


def ztn(value: int) -> int | None:
    return None if value == 0 else value
