from math import sqrt
import torch
from gtf_impl.utils import round_to_mult_of


class FromGTF:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("DIM", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[tuple[int, int]]:
        s = gtf.shape
        dimensions = tuple(map(int, (s[3], s[2])))
        return (dimensions, )


class Scale:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "dimensions": ("DIM", {}),
                "scale_width": ("FLOAT", {
                    "default": 1.0, "min": 0, "step": 0.001
                }),
                "scale_height": ("FLOAT", {
                    "default": 1.0, "min": 0, "step": 0.001
                }),
            },
        }

    RETURN_TYPES = ("DIM", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(
        dimensions: tuple[int, int],
        scale_width: float,
        scale_height: float
    ) -> tuple[tuple[int, int]]:
        width, height = dimensions
        new_width = int(width * scale_width)
        new_height = int(height * scale_height)
        return ((new_width, new_height), )


class ScaleToMegapixels:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "dimensions": ("DIM", {}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0, "step": 0.001
                }),
            },
        }

    RETURN_TYPES = ("DIM", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(
        dimensions: tuple[int, int],
        megapixels: float
    ) -> tuple[tuple[int, int]]:
        width, height = dimensions
        curr_megapixels = (width * height) / 1_000_000
        scale = sqrt(megapixels / curr_megapixels)
        dimensions = tuple(map(lambda x: int(x * scale), dimensions))
        return (dimensions, )


class Change:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "dimensions": ("DIM", {}),
                "delta_width": ("INT", {"default": 0}),
                "delta_height": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("DIM", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(
        dimensions: tuple[int, int],
        delta_width: int,
        delta_height: int
    ) -> tuple[tuple[int, int]]:
        width, height = dimensions
        new_width = width + delta_width
        new_height = height + delta_height
        return ((new_width, new_height), )


class AlignTo:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "dimensions": ("DIM", {}),
                "align_to": ("INT", {"default": 8, "min": 1}),
            },
        }

    RETURN_TYPES = ("DIM", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(
        dimensions: tuple[int, int],
        align_to: int
    ) -> tuple[tuple[int, int]]:
        width, height = dimensions
        new_width = round_to_mult_of(width, align_to)
        new_height = round_to_mult_of(height, align_to)
        return ((new_width, new_height), )


class FromRaw:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 1}),
                "height": ("INT", {"default": 1024, "min": 1}),
            },
        }

    RETURN_TYPES = ("DIM", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(width: int, height: int) -> tuple[tuple[int, int]]:
        return ((width, height), )


class ToRaw:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "dimensions": ("DIM", {}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    CATEGORY = "gtf/dimensions"
    FUNCTION = "f"

    @staticmethod
    def f(dimensions: tuple[int, int]) -> tuple[int, int]:
        width, height = dimensions
        return (width, height)
