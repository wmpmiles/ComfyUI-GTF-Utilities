"""Image filtering functions."""


import torch
import torch.signal as S
import torch.nn.functional as F
from gtf_impl.utils import gtf_min, gtf_max, invert


def blur_gaussian(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Preconditions:
    - sigma >= 0
    - tensor is in (B, C, H, W) dim order
    """
    channels = int(tensor.shape[1])
    radius = int(4.0 * sigma + 0.5)
    size = radius * 2 + 1
    pad = (radius, ) * 4
    gaussian = S.windows.gaussian(size, std=sigma)
    normalized_gaussian = gaussian / torch.sum(gaussian)
    kernel = \
        normalized_gaussian.reshape(1, 1, 1, -1).expand(channels, 1, -1, -1)
    tensor_padded = F.pad(tensor, pad, "reflect")
    tensor_blurred_w = \
        F.conv2d(tensor_padded, kernel, padding=0, groups=channels)
    kernel_permuted = kernel.permute(0, 1, 3, 2)
    tensor_blurred_wh = \
        F.conv2d(tensor_blurred_w, kernel_permuted, padding=0, groups=channels)
    return tensor_blurred_wh


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
