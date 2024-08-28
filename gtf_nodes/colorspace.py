import torch
from ..gtf_impl import colorspace as CS


class ColorspaceBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {}), }}

    RETURN_TYPES = ('GTF', )
    RETURN_NAMES = ('gtf', )
    CATEGORY = 'gtf/colorspace'
    FUNCTION = "f"


class SRGBGammaToLinear(ColorspaceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        linear = CS.srgb_gamma_to_linear(gtf)
        return (linear, )


class SRGBLinearToGamma(ColorspaceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        gamma = CS.srgb_linear_to_gamma(gtf)
        return (gamma, )


class LinearToLog(ColorspaceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        log = CS.linear_to_log(gtf, 0.0001)
        return (log, )


class LogToLinear(ColorspaceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        linear = CS.log_to_linear(gtf, 0.0001)
        return (linear, )


class StandardGammaToLinear(ColorspaceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        linear = CS.standard_gamma_to_linear(gtf)
        return (linear, )


class StandardLinearToGamma(ColorspaceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        linear = CS.standard_linear_to_gamma(gtf)
        return (linear, )


NODE_CLASS_MAPPINGS = {
    "GTF | Colorspace - SRGB Linear to Gamma":     SRGBLinearToGamma,
    "GTF | Colorspace - SRGB Gamma to Linear":     SRGBGammaToLinear,
    "GTF | Colorspace - Linear to Log":            LinearToLog,
    "GTF | Colorspace - Log to Linear":            LogToLinear,
    "GTF | Colorspace - Standard Linear to Gamma": StandardLinearToGamma,
    "GTF | Colorspace - Standard Gamma to Linear": StandardGammaToLinear,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
