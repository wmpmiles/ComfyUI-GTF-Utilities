"""
Colorspace related functions.
Reference: http://www.ericbrasseur.org/gamma.html?i=1#formulas
"""


import torch


def _quantize(tensor: torch.Tensor) -> torch.Tensor:
    COLOR_DEPTH = 16
    BASE = 2 ** COLOR_DEPTH - 1
    quantized = torch.round(tensor * BASE) / BASE
    return quantized


def srgb_gamma_to_linear(tensor: torch.Tensor):
    double = tensor.to(torch.float64)
    a = 0.055
    piece_lo = double / 12.92
    piece_hi = torch.pow((double + a) / (1 + a), 2.4)
    linear = torch.where(double <= 0.04045, piece_lo, piece_hi).to(torch.float)
    return linear


def srgb_linear_to_gamma(tensor: torch.Tensor):
    double = tensor.to(torch.float64)
    a = 0.055
    # We quantize the gamma space outputs to deal with floating-point
    # imprecision issues when truncation occurs
    piece_lo = _quantize(double * 12.92)
    piece_hi = _quantize((1 + a) * torch.pow(double, 1/2.4) - a)
    gamma = torch.where(double <= 0.0031308, piece_lo, piece_hi).to(torch.float)
    return gamma


def standard_gamma_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    linear = torch.pow(tensor.to(torch.float64), 2.2).to(torch.float)
    return linear


def standard_linear_to_gamma(tensor: torch.Tensor) -> torch.Tensor:
    linear = torch.pow(tensor.to(torch.float64), 1/2.2).to(torch.float)
    return linear


def linear_to_log(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    log = torch.log2(tensor.to(torch.float64) + eps).to(torch.float)
    return log


def log_to_linear(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    linear = (torch.exp2(tensor.to(torch.float64)) - eps).to(torch.float)
    return linear
