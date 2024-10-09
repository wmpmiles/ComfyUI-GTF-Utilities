"""Image filtering functions."""


import torch
import torch.nn.functional as F
from ..gtf_impl import utils as U
from ..gtf_impl import transform as TF
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
    kernel_expanded = kernel.expand(b, c, -1, -1)
    ph, pw = kh // 2, kw // 2
    padded_h = U.pad_tensor_reflect(tensor, 2, ph, ph)
    padded = U.pad_tensor_reflect(padded_h, 3, pw, pw)
    convolved = []
    for (b, k) in zip(padded, kernel_expanded):
        convolved += [F.conv2d(b.unsqueeze(0), k.unsqueeze(1), groups=c)]
    concatenated = torch.cat(convolved)
    return concatenated


def gradient_suppression_mask(
    norm: torch.Tensor, 
    angle: torch.Tensor
) -> torch.Tensor:
    padded = U.pad_tensor_reflect(U.pad_tensor_reflect(norm, 3, 1, 1), 2, 1, 1)
    views = []
    for i in range(9):
        x, y = i % 3, i // 3
        nx, ny = U.ztn(-(2-x)), U.ztn(-(2-y))
        views += [padded[:, :, y:ny, x:nx]]
    comparisons = []
    for i in range(4):
        lower = views[4] >= views[i]
        upper = views[4] >= views[8-i]
        comparisons += [torch.logical_and(lower, upper)]
    angle_index = (((angle + (9 * pi / 8)) * (4 / pi)).to(torch.int) + 3) % 4
    odd = (angle_index % 2).to(torch.bool)
    l = torch.where(odd, comparisons[1], comparisons[0])
    u = torch.where(odd, comparisons[3], comparisons[2])
    mask = torch.where((angle_index // 2).to(torch.bool), u, l)
    return mask


def hysteresis_threshold(weak: torch.Tensor, strong: torch.Tensor) -> torch.Tensor:
    b, c, h, w = weak.shape
    coloring, max_unique = TF.component_coloring(weak, True)
    strong_components = strong.to(torch.bool) * coloring
    batches = []
    for bi in range(b):
        channels = []
        for ci in range(c):
            strong_select = torch.zeros(max_unique + 1)
            strong_select[strong_components[bi, ci].flatten()] = 1
            strong_select[0] = 0
            result = strong_select.index_select(0, coloring[bi, ci].flatten()).reshape(h, w)
            channels += [result]
        batches += [torch.stack(channels)]
    thresholded = torch.stack(batches)
    return thresholded


# https://en.wikipedia.org/wiki/Otsu%27s_method
def otsus_method(gtf: torch.Tensor, bins: int) -> torch.Tensor:
    b, c, h, w = gtf.shape
    binned = (U.range_normalize(gtf, (2, 3)) * (bins - 1)).to(torch.int)
    batches = []
    for bi in range(b):
        channels = []
        for ci in range(c):
            n = torch.bincount(binned[bi, ci].flatten(), minlength=bins)
            n_cumsum = torch.cumsum(n, 0)
            N = n_cumsum[-1]
            omega = n_cumsum / N
            mu = torch.cumsum(n_cumsum * torch.arange(bins), 0) / N
            mu_T = mu[-1]
            sigma_2_B = (mu_T * omega - mu)**2 / (omega * (1 - omega))
            threshold = torch.argmax(sigma_2_B.nan_to_num()) / (bins - 1)
            if threshold.dim() == 1:
                return threshold[0]
            channels += [threshold.reshape(1, 1)]
        batches += [torch.stack(channels)]
    thresholds = torch.stack(batches)
    return thresholds


def patch_min(gtf: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    padded = U.pad_reflect_radius(gtf, (2, 3), radius)
    minimum = -F.max_pool2d(-padded, k, stride=1)
    return minimum


def patch_max(gtf: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    padded = U.pad_reflect_radius(gtf, (2, 3), radius)
    maximum = F.max_pool2d(padded, k, stride=1)
    return maximum


def patch_median(gtf: torch.Tensor, radius: int) -> torch.Tensor:
    b, c, h, w = gtf.shape
    k = 2 * radius + 1
    padded = U.pad_reflect_radius(gtf, (2, 3), radius)
    unfolded = U.unfold(padded, k, k).flatten(-2)
    median = unfolded.median(dim=-1, keepdim=False).values
    return median


def patch_range_normalize(gtf: torch.Tensor, radius: int) -> torch.Tensor: 
    r = radius 
    k = 2 * r + 1
    minimum = patch_min(gtf, radius)
    ln = gtf - minimum
    maximum = patch_max(ln, radius)
    normalized = ln / maximum
    return normalized

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


def open(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    opened = dilate(erode(tensor, radius), radius)
    return opened


#               #
# OTHER FILTERS #
#               #



def quantize(
    tensor: torch.Tensor,
    steps: int,
    mode: Literal["round", "floor", "ceiling"],
) -> torch.Tensor:
    F_MAP = {
        "round": torch.round,
        "floor": torch.floor,
        "ceiling": torch.ceil,
    }
    if mode not in F_MAP:
        raise ValueError("`mode` must be one of ('round', 'floor', 'ceiling')")
    max_val = steps - 1
    quantized = F_MAP[mode](tensor * max_val) / max_val
    return quantized
