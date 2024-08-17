import torch
from gtf_impl import colorspace as cs


class SRGBGammaToLinear:
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
        linear = cs.srgb_gamma_to_linear(gtf)
        return (linear, )


class SRGBLinearToGamma:
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
        gamma = cs.srgb_linear_to_gamma(gtf)
        return (gamma, )
