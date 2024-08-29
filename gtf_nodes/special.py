import torch
from ..gtf_impl import special as SP
from math import ceil


# BASE CLASSES

class SpecialBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {})}}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math/special"
    FUNCTION = "f"


# NODES

class Sinc(SpecialBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        sinc = torch.sinc(gtf)
        return (sinc, )


class Jinc(SpecialBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        jinc = SP.jinc(gtf)
        return (jinc, )


class Gaussian(SpecialBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "sigma": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, sigma: float) -> tuple[torch.Tensor]:
        gaussian = SP.gaussian(gtf, sigma)
        return (gaussian, )


class DerivativeOfGaussian(SpecialBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "sigma": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, sigma: float) -> tuple[torch.Tensor]:
        dog = SP.derivative_of_gaussian(gtf, sigma)
        return (dog, )


class GaussianAreaRadius(SpecialBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "sigma": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
            "area": ("FLOAT", {"default": 0.99, "min": 0, "max": 1, "step": 0.001}),
        }}

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("radius", )

    @staticmethod
    def f(sigma: float, area: float) -> tuple[int]:
        radius = int(ceil(SP.gaussian_area_radius(sigma, area)))
        return (radius, )


class DerivativeOfGaussianAreaRadius(SpecialBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "sigma": ("FLOAT", {"default": 1, "min": 0.001, "step": 0.001}),
            "area": ("FLOAT", {"default": 0.99, "min": 0, "max": 1, "step": 0.001}),
        }}

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("radius", )

    @staticmethod
    def f(sigma: float, area: float) -> tuple[int]:
        radius = int(ceil(SP.derivative_of_gaussian_area_radius(sigma, area)))
        return (radius, )


NODE_CLASS_MAPPINGS = {
    "GTF | Special - Sinc": Sinc,
    "GTF | Special - Jinc": Jinc,
    "GTF | Special - Gaussian": Gaussian,
    "GTF | Special - Derivative of Gaussian": DerivativeOfGaussian,
    "GTF | Helper - Gaussian Area Radius": GaussianAreaRadius,
    "GTF | Helper - Derivative of Gaussian Area Radius": DerivativeOfGaussianAreaRadius,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
