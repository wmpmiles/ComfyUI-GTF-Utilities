import torch
from typing import Any
from .colorspace import srgb_gamma_to_linear, srgb_linear_to_gamma


class ColorspaceSRGBGammaToLinear:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/colorspace"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        linear = srgb_gamma_to_linear(tensor)
        return ((linear, typeinfo, extra), )


class ColorspaceSRGBLinearToGamma:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/colorspace"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        linear = srgb_linear_to_gamma(tensor)
        return ((linear, typeinfo, extra), )
