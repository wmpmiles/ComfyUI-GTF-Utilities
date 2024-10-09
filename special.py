import torch
from torch import Tensor
from math import sqrt, log, pi
from scipy.special import erfinv


# FUNCTIONS

def jinc(x: Tensor) -> Tensor:
    p0: Tensor = (2 / pi) * torch.special.bessel_j1(pi * x) / x
    j = p0.where(x != 0, 1)
    return j


def gaussian(x: Tensor, sigma: float) -> Tensor:
    coef = 1 / (sigma * sqrt(2 * pi))
    g = coef * torch.exp(x**2 / (-2 * sigma**2))
    return g


def derivative_of_gaussian(x: Tensor, sigma: float) -> Tensor:
    coef = -1 / (sigma**3 * sqrt(2 * pi))
    dog = coef * x * torch.exp(x**2 / (-2 * sigma**2))
    return dog


# HELPERS

def gaussian_area_radius(sigma: float, area: float) -> float:
    if area <= 0 or area >= 1:
        raise ValueError("`area` must be in (0, 1).")
    radius = sqrt(2) * sigma * erfinv(area)
    return radius


def derivative_of_gaussian_area_radius(sigma: float, area: float) -> float:
    if area <= 0 or area >= 1:
        raise ValueError("`area` must be in (0, 1).")
    radius = sqrt(2) * sigma * sqrt(log(1 / (1 - area)))
    return radius
