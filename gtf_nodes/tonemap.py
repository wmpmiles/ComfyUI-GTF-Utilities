import torch
from gtf_impl import tonemap as TM


class Reinhard:
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
        tonemapped = TM.reinhard(gtf)
        return (tonemapped, )


class ReinhardExtended:
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
        tonemapped = TM.reinhard_extended(gtf, gtf_whitepoint)
        return (tonemapped, )


class ReinhardLuminance:
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
        tonemapped = TM.reinhard_luminance(gtf, gtf_luminance)
        return (tonemapped, )


class ReinhardExtendedLuminance:
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
        tonemapped = TM.reinhard_extended_luminance(
            gtf,
            gtf_luminance,
            gtf_whitepoint,
        )
        return (tonemapped, )


class ReinhardJodie:
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
        tonemapped = TM.reinhard_jodie(gtf, gtf_luminance)
        return (tonemapped, )


class ReinhardJodieExtended:
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
        tonemapped = TM.reinhard_jodie_extended(
            gtf,
            gtf_luminance,
            gtf_whitepoint
        )
        return (tonemapped, )


class Uncharted2:
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
        tonemapped = TM.uncharted_2(gtf)
        return (tonemapped, )


class ACES:
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
        tonemapped = TM.aces(gtf)
        return (tonemapped, )
