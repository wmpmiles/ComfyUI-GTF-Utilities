import torch
from ..impl.tonemap import *


class TonemapReinhard:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = tonemap_reinhard(gtf)
        return (tonemapped, )


class TonemapReinhardExtended:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "gtf_whitepoint": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        gtf_whitepoint: torch.Tensor
    ) -> tuple[torch.Tensor]:
        tonemapped = tonemap_reinhard_extended(gtf, gtf_whitepoint)
        return (tonemapped, )


class TonemapReinhardLuminance:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "luminance": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        gtf_luminance: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        tonemapped = tonemap_reinhard_luminance(gtf, gtf_luminance)
        return (tonemapped, )


class TonemapReinhardExtendedLuminance:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "gtf_luminance": ("GTF", {}),
                "gtf_whitepoint": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        gtf_luminance: torch.Tensor, 
        gtf_whitepoint: torch.Tensor, 
    ) -> tuple[torch.Tensor]:
        tonemapped = tonemap_reinhard_extended_luminance(
            gtf, 
            gtf_luminance, 
            gtf_whitepoint,
        )
        return (tonemapped, )


class TonemapReinhardJodie:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "luminance": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        gtf_luminance: torch.Tensor
    ) -> tuple[torch.Tensor]:
        tonemapped = tonemap_reinhard_jodie(gtf, gtf_luminance)
        return (tonemapped, )


class TonemapReinhardJodieExtended:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "gtf_luminance": ("GTF", {}),
                "gtf_whitepoint": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        gtf_luminance: torch.Tensor, 
        gtf_whitepoint: torch.Tensor, 
    ) -> tuple[torch.Tensor]:
        tonemapped = tonemap_reinhard_jodie_extended(
            gtf, 
            gtf_luminance, 
            gtf_whitepoint
        )
        return (tonemapped, )


class TonemapUncharted2:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = tonemap_uncharted_2(gtf)
        return (tonemapped, )


class TonemapACES:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/tonemap"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tonemapped = tonemap_aces(gtf)
        return (tonemapped, )

