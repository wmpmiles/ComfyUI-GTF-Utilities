import torch
from ..gtf_impl import tonemap as TM


# BASE CLASSES

class TonemapBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {})}}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"


# NODES

class Reinhard(TonemapBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.reinhard(gtf)
        return (tonemapped, )


class ReinhardExtended(TonemapBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_whitepoint": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_whitepoint: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.reinhard_extended(gtf, gtf_whitepoint)
        return (tonemapped, )


class ReinhardLuminance(TonemapBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
                "gtf": ("GTF", {}),
                "gtf_luminance": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_luminance: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.reinhard_luminance(gtf, gtf_luminance)
        return (tonemapped, )


class ReinhardExtendedLuminance(TonemapBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_luminance": ("GTF", {}),
            "gtf_whitepoint": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_luminance: torch.Tensor, gtf_whitepoint: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.reinhard_extended_luminance(gtf, gtf_luminance, gtf_whitepoint)
        return (tonemapped, )


class ReinhardJodie(TonemapBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_luminance": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_luminance: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.reinhard_jodie(gtf, gtf_luminance)
        return (tonemapped, )


class ReinhardJodieExtended(TonemapBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_luminance": ("GTF", {}),
            "gtf_whitepoint": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_luminance: torch.Tensor, gtf_whitepoint: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.reinhard_jodie_extended(gtf, gtf_luminance, gtf_whitepoint)
        return (tonemapped, )


class Uncharted2(TonemapBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.uncharted_2(gtf)
        return (tonemapped, )


class ACES(TonemapBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = TM.aces(gtf)
        return (tonemapped, )


NODE_CLASS_MAPPINGS = {
    "GTF | Tonemap - Reinhard":                         Reinhard,
    "GTF | Tonemap - Reinhard Extended":                ReinhardExtended,
    "GTF | Tonemap - Reinhard over Luminance":          ReinhardLuminance,
    "GTF | Tonemap - Reinhard Extended over Luminance": ReinhardExtendedLuminance,
    "GTF | Tonemap - Reinhard-Jodie":                   ReinhardJodie,
    "GTF | Tonemap - Reinhard-Jodie Extended":          ReinhardJodieExtended,
    "GTF | Tonemap - Uncharted 2":                      Uncharted2,
    "GTF | Tonemap - ACES":                             ACES,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
