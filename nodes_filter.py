import torch
from typing import Any
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
    def f(gtf: tuple[torch.Tensor, str, Any], operation: str, radius: int) -> tuple[tuple[torch.Tensor, str, Any]]:
        if radius == 0:
            return (gtf, )
        tensor, typeinfo, extra = gtf
        match operation:
            case "dilate":
                filtered = tensor_dilate(tensor, radius)
            case "erode":
                filtered = tensor_erode(tensor, radius)
            case "open":
                filtered = tensor_open(tensor, radius)
            case "close":
                filtered = tensor_close(tensor, radius)
        return ((filtered, typeinfo, extra), )


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
    def f(gtf: tuple[torch.Tensor, str, Any], sigma: float) -> tuple[tuple[torch.Tensor, str, Any]]:
        if sigma <= 0.0:
            return (gtf, )
        tensor, typeinfo, extra = gtf
        blurred = blur_gaussian(tensor, sigma)
        return ((blurred, typeinfo, extra), )
