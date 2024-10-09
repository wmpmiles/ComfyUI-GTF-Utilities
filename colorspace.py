"""
Colorspace related functions.
Reference: http://www.ericbrasseur.org/gamma.html?i=1#formulas
"""


import torch
from torch import Tensor


def srgb_gamma_to_linear(tensor: Tensor) -> Tensor:
    double = tensor.to(torch.float64)
    a = 0.055
    piece_lo = double / 12.92
    piece_hi = torch.pow((double + a) / (1 + a), 2.4)
    linear = torch.where(double <= 0.04045, piece_lo, piece_hi).to(torch.float)
    return linear


def srgb_linear_to_gamma(tensor: Tensor) -> Tensor:
    double = tensor.to(torch.float64)
    a = 0.055
    piece_lo = double * 12.92
    piece_hi = (1 + a) * torch.pow(double, 1/2.4) - a
    gamma = torch.where(double <= 0.0031308, piece_lo, piece_hi).to(torch.float)
    return gamma


def standard_gamma_to_linear(tensor: Tensor) -> Tensor:
    linear = torch.pow(tensor.to(torch.float64), 2.2).to(torch.float)
    return linear


def standard_linear_to_gamma(tensor: Tensor) -> Tensor:
    linear = torch.pow(tensor.to(torch.float64), 1/2.2).to(torch.float)
    return linear


def linear_to_log(tensor: Tensor, eps: float) -> Tensor:
    log = torch.log2(tensor.to(torch.float64) + eps).to(torch.float)
    return log


def log_to_linear(tensor: Tensor, eps: float) -> Tensor:
    linear = (torch.exp2(tensor.to(torch.float64)) - eps).to(torch.float)
    return linear
