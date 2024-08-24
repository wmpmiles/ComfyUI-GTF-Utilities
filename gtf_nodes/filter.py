import torch
from gtf_impl import filter as F
from gtf_impl import utils as U


class Invert:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        inverted = U.invert(gtf)
        return (inverted, )


class NormalizeKernel:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        normalized = U.normalize_kernel(gtf)
        return (normalized, )


class Convolve:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
                "gtf_kernel": ("GTF", {})
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, gtf_kernel: torch.Tensor) -> tuple[torch.Tensor]:
        convolved = F.convolve_2d(gtf, gtf_kernel)
        return (convolved, )


class MorphologicalFilter:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
                "operation": (["dilate", "erode", "open", "close"], {}),
                "radius": ("INT", {"default": 3, "min": 1})
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor,
        operation: str,
        radius: int
    ) -> tuple[torch.Tensor]:
        if radius == 0:
            return (gtf, )
        match operation:
            case "dilate":
                filtered = F.dilate(gtf, radius)
            case "erode":
                filtered = F.erode(gtf, radius)
            case "open":
                filtered = F.open(gtf, radius)
            case "close":
                filtered = F.close(gtf, radius)
        return (filtered, )


class KernelGaussian:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "sigma": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter/kernel"
    FUNCTION = "f"

    @staticmethod
    def f(sigma: float) -> tuple[torch.Tensor]:
        kernel = F.kernel_gaussian(sigma)
        return (kernel, )
