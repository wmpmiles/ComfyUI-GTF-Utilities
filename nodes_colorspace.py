import torch
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
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        linear = srgb_gamma_to_linear(gtf)
        return (linear, )


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
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        gamma = srgb_linear_to_gamma(gtf)
        return (gamma, )
