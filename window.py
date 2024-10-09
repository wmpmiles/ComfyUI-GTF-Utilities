import torch


# FUNCTIONS

def boxcar(x: torch.Tensor, radius: int) -> torch.Tensor:
    windowed = (torch.abs(x) <= radius).to(torch.float)
    return windowed


def triangle(x: torch.Tensor, radius: int) -> torch.Tensor:
    windowed = ((radius - torch.abs(x)) / radius).clamp(0, 1)
    return windowed


def lanczos(x: torch.Tensor, radius: int) -> torch.Tensor:
    windowed = torch.sinc(x / radius) * (torch.abs(x) <= radius)
    return windowed


def mitchell_netravali(x: torch.Tensor, b: float, c: float) -> torch.Tensor:
    x_abs = torch.abs(x)
    x_abs2 = x_abs * x_abs
    x_abs3 = x_abs2 * x_abs
    p0 = (12 + -9*b + -6*c)*x_abs3 + (-18 + 12*b + 6*c)*x_abs2 + (6 + -2*b)
    p1 = (-b + -6*c)*x_abs3 + (6*b + 30*c)*x_abs2 + (-12*b + -48*c)*x_abs + (8*b + 24*c)
    p0_mask = x_abs < 1
    p1_mask = torch.logical_and(torch.logical_not(p0_mask), x_abs < 2)
    p = p0 * p0_mask + p1 * p1_mask
    mn = p / 6
    return mn


def area(x: torch.Tensor, l: int, L: int) -> torch.Tensor:
    l, L = (l, L) if L <= l else (L, l)
    windowed = (((L + l) - (2 * l) * torch.abs(x)) / (2 * L)).clamp(0, 1)
    return windowed


# HELPERS

def mitchell_netravali_radius() -> int:
    return 2


def area_window_radius() -> int:
    return 1
