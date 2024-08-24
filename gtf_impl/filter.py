"""Image filtering functions."""


import torch
import torch.signal as S
import torch.nn.functional as F
import gtf_impl.utils as U


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
    window_offsets = U.outer_sum(torch.arange(kh), torch.arange(kw))
    indices_2d = torch.arange(h * w).reshape(1, 1, oh, ow, 1, 1)
    indices_4d = indices_2d + window_offsets.reshape(1, 1, 1, 1, kh, kw)
    indices_1d = indices_4d.reshape(1, 1, -1)
    data_1d = tensor.reshape(b, c, -1).index_select(2, indices_1d)
    data_4d = data_1d.reshape(b, c, oh, ow, kh, kw)
    return data_4d


def kernel_gaussian(sigma: float) -> torch.Tensor:
    """
    Preconditions:
    - sigma >= 0
    - tensor is in (B, C, H, W) dim order
    """
    if sigma < 0:
        raise ValueError("Sigma must be greater than or equal to 0.")
    radius = int(4.0 * sigma + 0.5)
    size = radius * 2 + 1
    gaussian = S.windows.gaussian(size, std=sigma)
    reshaped = gaussian.reshape(1, 1, 1, -1)
    return reshaped


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
    eroded = invert(dilate(invert(tensor), radius))
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

def stretch_contrast(
    tensor: torch.Tensor,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> torch.Tensor:
    cur_min = gtf_min(tensor, (2, 3))
    minned = tensor - (cur_min - min_val)
    cur_max = gtf_max(minned, (2, 3))
    maxxed = minned * (max_val / cur_max)
    clamped = maxxed.clamp(min_val, max_val)
    return clamped
