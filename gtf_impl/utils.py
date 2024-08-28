import torch
from typing import Iterable
from math import pi, sqrt


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



# RAW FUNCTIONS

def jinc(x: torch.Tensor) -> torch.Tensor:
    p0: torch.Tensor = (2 / pi) * torch.special.bessel_j1(pi * x) / x
    j = p0.where(x != 0, 1)
    return j


def gaussian(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_r = torch.reciprocal(sigma)
    coef = sigma_r / sqrt(2 * pi)
    g = coef * torch.exp(-0.5 * x**2 * sigma_r**2)
    return g


def derivative_of_gaussian(
    x: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    sigma_r = torch.reciprocal(sigma)
    coef = sigma_r**3 / sqrt(2 * pi)
    dog = coef * x * torch.exp(-0.5 * x**2 / sigma_r**2)
    return dog
