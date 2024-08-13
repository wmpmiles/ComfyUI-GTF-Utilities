"""Image filtering functions."""


import torch
import torch.signal as S
import torch.nn.functional as F
from .utils import gtf_max, gtf_min, tensor_invert


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
    kernel = normalized_gaussian.reshape(1, 1, 1, -1).expand(channels, 1, -1, -1)
    tensor_padded = F.pad(tensor, pad, "reflect")
    tensor_blurred_w = F.conv2d(tensor_padded, kernel, padding=0, groups=channels)
    kernel_permuted = kernel.permute(0, 1, 3, 2)
    tensor_blurred_wh = F.conv2d(tensor_blurred_w, kernel_permuted, padding=0, groups=channels)
    return tensor_blurred_wh


###                       ###
### MORPHOLOGICAL FILTERS ###
###                       ###

# precondition: radius > 0
def tensor_dilate(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Preconditions:
    - radius > 0
    - tensor is in (B, C, H, W) dim order
    """
    kernel_size = 2 * radius - 1
    padding = radius - 1
    dilated = torch.nn.functional.max_pool2d(tensor, kernel_size, stride=1, padding=padding)
    return dilated


def tensor_erode(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Preconditions:
    - radius > 0
    - tensor is in (B, C, H, W) dim order
    """
    eroded = tensor_invert(tensor_dilate(tensor_invert(tensor), radius))
    return eroded


def tensor_close(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Preconditions:
    - radius > 0
    - tensor is in (B, C, H, W) dim order
    """
    closed = tensor_erode(tensor_dilate(tensor, radius), radius)
    return closed


def tensor_open(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Preconditions:
    - radius > 0
    - tensor is in (B, C, H, W) dim order
    """
    opened = tensor_dilate(tensor_erode(tensor, radius), radius)
    return opened

###               ###
### OTHER FILTERS ###
###               ###

def stretch_contrast(tensor: torch.Tensor, min: float = 0.0, max: float = 1.0) -> torch.Tensor:
    cur_min = gtf_min(tensor)
    minned = tensor - (cur_min - min)
    cur_max = gtf_max(minned)
    maxxed = minned * (max / cur_max)
    clamped = maxxed.clamp(min, max)
    return clamped

