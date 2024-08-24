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


class BinaryThreshold:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
                "threshold": ("FLOAT", {"default": 0.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, threshold: float) -> tuple[torch.Tensor]:
        thresholded = (gtf >= threshold).to(torch.float)
        return (thresholded, )


class Quantize:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
                "steps": ("INT", {"default": 256, "min": 2, "max": 1_000_000}),
                "mode": ([*F.F_MAP.keys()], ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, steps: int, mode: str) -> tuple[torch.Tensor]:
        quantized = F.quantize(gtf, steps, mode)
        return (quantized, )


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
