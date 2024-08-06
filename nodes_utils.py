from math import sqrt
from typing import Any
import torch


class Dimensions:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    CATEGORY = "gtf/utils"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[int, int]:
        tensor, _, _ = gtf
        s = tensor.shape
        dimensions = tuple(map(int, (s[3], s[2])))
        return dimensions


class ScaleDimensions:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "width": ("INT", {}),
                "height": ("INT", {}),
                "scale": ("FLOAT", { "default": 1.0, "min": 0, "step": 0.001 }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    CATEGORY = "gtf/utils"
    FUNCTION = "f"

    @staticmethod
    def f(width: int, height: int, scale: float) -> tuple[int, int]:
        dimensions = tuple(map(lambda x: int(x * scale), (width, height)))
        return dimensions


class ScaleDimensionsToMegapixels:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "width": ("INT", {}),
                "height": ("INT", {}),
                "megapixels": ("FLOAT", { "default": 1.0, "min": 0, "step": 0.001 }),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    CATEGORY = "gtf/utils"
    FUNCTION = "f"


    @staticmethod
    def f(width: int, height: int, megapixels: float) -> tuple[int, int]:
        curr_megapixels = (width * height) / 1_000_000
        scale = sqrt(megapixels / curr_megapixels)
        dimensions = tuple(map(lambda x: int(x * scale), (width, height)))
        return dimensions
