import torch
from math import sqrt, log, pi
from scipy.special import erfinv


# FUNCTIONS

def jinc(x: torch.Tensor) -> torch.Tensor:
    p0: torch.Tensor = (2 / pi) * torch.special.bessel_j1(pi * x) / x
    j = p0.where(x != 0, 1)
    return j


def gaussian(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_r = torch.reciprocal(sigma)
    coef = sigma_r / sqrt(2 * pi)
    g = coef * torch.exp(-0.5 * x**2 * sigma_r**2)
    return g


def derivative_of_gaussian(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_r = torch.reciprocal(sigma)
    coef = sigma_r**3 / sqrt(2 * pi)
    dog = coef * x * torch.exp(-0.5 * x**2 / sigma_r**2)
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
