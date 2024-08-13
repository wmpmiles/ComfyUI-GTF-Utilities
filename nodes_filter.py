import torch
from .filter import blur_gaussian, tensor_close, tensor_dilate, tensor_erode, tensor_open


class MorphologicalFilter:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "operation": (["dilate", "erode", "open", "close"], {}),
                "radius": ("INT", { "default": 3, "min": 1 })
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, operation: str, radius: int) -> tuple[torch.Tensor]:
        if radius == 0:
            return (gtf, )
        match operation:
            case "dilate":
                filtered = tensor_dilate(gtf, radius)
            case "erode":
                filtered = tensor_erode(gtf, radius)
            case "open":
                filtered = tensor_open(gtf, radius)
            case "close":
                filtered = tensor_close(gtf, radius)
        return (filtered, )


class BlurGaussian:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "sigma": ("FLOAT", { "default": 3.0, "min": 0.0, "step": 0.1 }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/filter"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, sigma: float) -> tuple[torch.Tensor]:
        if sigma <= 0.0:
            return (gtf, )
        blurred = blur_gaussian(gtf, sigma)
        return (blurred, )
