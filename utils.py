import torch
from torch import Tensor
from typing import Iterable


def round_up_to_mult_of(number: int, mult_of: int) -> int:
    aligned = ((number + mult_of - 1) // mult_of) * mult_of
    return aligned


def gtf_min(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    curr = tensor
    for dim in dims:
        curr = torch.amin(curr, dim, True)
    return curr


def gtf_max(tensor: Tensor, dims: Iterable[int]) -> Tensor:
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


def invert(tensor: Tensor) -> Tensor:
    inverted = (1.0 - tensor)
    return inverted


def pad_tensor_reflect(tensor: Tensor, dim: int, pad_start: int, pad_end: int) -> Tensor:
    length = tensor.shape[dim]
    start = tensor[slice_dim(dim, 1, pad_start+1)].flip(dim)
    end = tensor[slice_dim(dim, length-(pad_end+1), -1)].flip(dim)
    padded = torch.cat((start, tensor, end), dim)
    return padded


def pad_reflect_radius(tensor: Tensor, dims: tuple[int], radius: int) -> Tensor:
    padded = tensor
    for dim in dims:
        padded = pad_tensor_reflect(padded, dim, radius, radius)
    return padded


def unfold(gtf: Tensor, kh: int, kw: int) -> Tensor:
    unfolded = gtf.unfold(3 , kw, 1).unfold(2, kh, 1)
    return unfolded


def outer_sum(lhs: Tensor, rhs: Tensor) -> Tensor:
    ret = lhs.unsqueeze(1) + rhs.unsqueeze(0)
    return ret


def sum_normalize(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    dims = sorted(dims)
    denominator = tensor
    for dim in reversed(dims):
        denominator = denominator.sum(dim)
    for dim in dims:
        denominator = denominator.unsqueeze(dim)
    normalized = tensor / denominator
    return normalized


def range_normalize(tensor: Tensor, dims: Iterable[int]) -> Tensor:
    cur_min = gtf_min(tensor, dims)
    minned = tensor - cur_min
    cur_max = gtf_max(minned, dims)
    maxxed = minned / cur_max
    clamped = maxxed.clamp(0, 1)
    return clamped


def ztn(value: int) -> int | None:
    return None if value == 0 else value


def dimensions_scale_to_megapixels(width: int, height: int, megapixels: float) -> tuple[int, int]:
    curr_mp = (width * height) / 1_000_000
    scale = sqrt(megapixels / curr_mp)
    nw, nh = int(width * scale), int(height * scale)
    return (nw, nh)


def gtf_cartesian_to_polar(gtf_x: Tensor, gtf_y: Tensor) -> tuple[Tensor, Tensor]:
    r = torch.hypot(gtf_x, gtf_y)
    theta = torch.atan2(gtf_y, gtf_x).nan_to_num()
    return (r, theta)


def gtf_polar_to_cartesian(gtf_r: Tensor, gtf_theta: Tensor) -> tuple[Tensor, Tensor]:
    x = gtf_r * torch.cos(gtf_theta)
    y = gtf_r * torch.sin(gtf_theta)
    return (x, y)
    

def gtf_rgb(r: int, g: int, b: int) -> Tensor:
    rgb = (Tensor((r, g, b)).to(torch.float) / 255).reshape(1, 3, 1, 1)
    return rgb
