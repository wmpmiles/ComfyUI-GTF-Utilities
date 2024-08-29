"""Image filtering functions."""


import torch
import torch.nn.functional as F
from ..gtf_impl import utils as U
from typing import Literal
from math import pi


def convolve_2d(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = tensor.shape
    kb, kc, kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernels must have odd height and width.")
    if kb != 1 and kb != b:
        raise ValueError("A kernel with batch size greater than 1 must have \
                         the same batch size as the GTF.")
    if kc != 1 and kc != c:
        raise ValueError("A kernel with more than 1 channel must have the \
                         same number of chdannels as the GTF.")
    ph, pw = kh // 2, kw // 2
    kernel_reshaped = kernel.reshape(kb, kc, 1, 1, kh, kw)
    padded_h = U.pad_tensor_reflect(tensor, 2, ph, ph)
    padded = U.pad_tensor_reflect(padded_h, 3, pw, pw)
    unfolded = _unfold_2d(padded, kh, kw)
    multiplied = unfolded * kernel_reshaped
    convolved = multiplied.sum(-1).sum(-1)
    return convolved


def _unfold_2d(tensor: torch.Tensor, kh: int, kw: int) -> torch.Tensor:
    b, c, h, w = tensor.shape
    oh, ow = h - kh + 1, w - kw + 1
    window_offsets = U.outer_sum(torch.arange(kh) * w, torch.arange(kw))
    indices_2d = U.outer_sum(torch.arange(oh) * w, torch.arange(ow))
    indices_4d = window_offsets.reshape(1, 1, kh, kw) \
        + indices_2d.reshape(oh, ow, 1, 1)
    data_1d = tensor.reshape(b, c, -1).index_select(2, indices_4d.flatten())
    data_4d = data_1d.reshape(b, c, oh, ow, kh, kw)
    return data_4d


def gradient_suppression(
    norm: torch.Tensor, 
    angle: torch.Tensor
) -> torch.Tensor:
    arg_normalized = (angle + pi) / (2 * pi)
    offset_a = torch.round(arg_normalized * 8).to(torch.int) % 8
    offset_b = (offset_a + 4) % 8
    indices = torch.stack((indices_a, indices_b))
    window_order = [3, 6, 7, 8, 4, 2, 1, 0]
    unwrapped = _unwrap_2d(norm, window_order)
    data = 1


def _unwrap_2d(tensor: torch.Tensor, window_order: list[int]) -> torch.Tensor:
    # Assumes (1, 1, 1, 1) padding
    b, c, h, w = tensor.shape
    oh, ow = h-2, w-2
    window_offsets = torch.tensor([0, 1, 2, w, w+1, w+2, 2*w, 2*w+1, 2*w+2])
    window_order = torch.tensor(window_order, dtype=torch.int)
    permuted_offsets = window_offsets.index_select(0, window_order)
    indices = U.outer_sum(torch.arange(oh) * w, torch.arange(ow)).flatten()
    indices_2d = U.outer_sum(indices, permuted_offsets)
    data_1d = tensor.reshape(b, c, -1).index_select(2, indices_2d.flatten())
    data_3d = data_1d.reshape(b, c, oh, ow, 8)
    return data_3d


#                       #
# MORPHOLOGICAL FILTERS #
#                       #

# precondition: radius > 0
def dilate(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    kernel_size = 2 * radius - 1
    padding = radius - 1
    dilated = F.max_pool2d(tensor, kernel_size, stride=1, padding=padding)
    return dilated


def erode(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    eroded = U.invert(dilate(U.invert(tensor), radius))
    return eroded


def close(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    closed = erode(dilate(tensor, radius), radius)
    return closed


def tensor_open(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    opened = dilate(erode(tensor, radius), radius)
    return opened


#               #
# OTHER FILTERS #
#               #

F_MAP = {
    "round": torch.round,
    "floor": torch.floor,
    "ceiling": torch.ceil,
}


def quantize(
    tensor: torch.Tensor,
    steps: int,
    mode: Literal["round", "floor", "ceiling"],
) -> torch.Tensor:
    max_val = steps - 1
    quantized = F_MAP[mode](tensor * max_val) / max_val
    return quantized
