import torch
import torch.nn.functional as F
from .resample import slice_dim


def pad_dim(tensor: torch.Tensor, dim: int, pad: tuple[int, int]) -> torch.Tensor:
    dims = tensor.dim()
    pad_list = [0, 0] * (dims - dim - 1) + list(pad)
    padded = F.pad(tensor, pad_list)
    return padded


def pad_crop_dim(tensor: torch.Tensor, dim: int, deltas: tuple[int, int]) -> torch.Tensor:
    dl, dr = deltas
    sl, sr = (min(0, dl), min(0, dr))
    pl, pr = (max(0, dl), max(0, dr))
    cropped = tensor[slice_dim(dim, -sl, sr if sr else None)]
    padded = pad_dim(cropped, dim, (pl, pr))
    return padded


def crop_uncrop(tensor: torch.Tensor, dim: int, new_length: int, anchor: str) -> torch.Tensor:
    length = int(tensor.shape[dim])
    delta = new_length - length
    match anchor:
        case "left": deltas = (0, delta)
        case "right": deltas = (delta, 0)
        case "middle": deltas = (delta // 2, delta - delta // 2)
        case _: raise ValueError("anchor must be one of [left, right, middle]")
    pad_cropped = pad_crop_dim(tensor, dim, deltas)
    return pad_cropped