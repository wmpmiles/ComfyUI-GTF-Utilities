"""Colorspace related functions."""


import torch


#http://www.ericbrasseur.org/gamma.html?i=1#formulas
def srgb_gamma_to_linear(tensor):
    """Precondition: `tensor` is a floating point torch tensor containing only sRGB values in gamma space."""
    a = 0.055
    piece_lo = tensor / 12.92
    piece_hi = torch.pow((tensor + a) / (1 + a), 2.4)
    tensor = torch.where(tensor <= 0.04045, piece_lo, piece_hi)
    return tensor

def srgb_linear_to_gamma(tensor):
    """Precondition: `tensor` is a floating point torch tensor containing only sRGB values in linear space."""
    a = 0.055
    piece_lo = tensor * 12.92
    piece_hi = (1 + a) * torch.pow(tensor, 1/2.4) - a
    tensor = torch.where(tensor <= 0.0031308, piece_lo, piece_hi)
    return tensor
