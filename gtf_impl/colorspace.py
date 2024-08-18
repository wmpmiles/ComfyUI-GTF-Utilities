"""
Colorspace related functions.
Reference: http://www.ericbrasseur.org/gamma.html?i=1#formulas
"""


import torch


def srgb_gamma_to_linear(tensor: torch.Tensor):
    a = 0.055
    piece_lo = tensor / 12.92
    piece_hi = torch.pow((tensor + a) / (1 + a), 2.4)
    tensor = torch.where(tensor <= 0.04045, piece_lo, piece_hi)
    return tensor


def srgb_linear_to_gamma(tensor: torch.Tensor):
    a = 0.055
    piece_lo = tensor * 12.92
    piece_hi = (1 + a) * torch.pow(tensor, 1/2.4) - a
    tensor = torch.where(tensor <= 0.0031308, piece_lo, piece_hi)
    return tensor


def linear_to_log(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    log = torch.log2(tensor + eps)
    return log


def log_to_linear(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    linear = torch.exp2(tensor) - eps
    return linear
