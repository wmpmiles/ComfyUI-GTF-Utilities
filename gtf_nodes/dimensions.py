from math import sqrt
import torch
from .. import types as T
from ..gtf_impl import utils as U


class DimensionsBase:
    RETURN_TYPES = ("DIMENSIONS", )
    RETURN_NAMES = ("dimensions", )
    CATEGORY =  'gtf/dimensions'
    FUNCTION = "f"


class FromGTF(DimensionsBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {'gtf': ('GTF', {})}}

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[T.Dimensions]:
        _, _, h, w = gtf.shape
        dimensions = (w, h)
        return (dimensions, )


class Scale(DimensionsBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "dimensions": ("DIMENSIONS", {}),
            "scale_width": ("FLOAT", {"default": 1.0, "min": 0, "step": 0.001}),
            "scale_height": ("FLOAT", {"default": 1.0, "min": 0, "step": 0.001}),
        }}

    @staticmethod
    def f(dimensions: T.Dimensions, scale_width: float, scale_height: float ) -> tuple[T.Dimensions]:
        width, height = dimensions
        new_width = int(width * scale_width)
        new_height = int(height * scale_height)
        return ((new_width, new_height), )


class ScaleToMegapixels(DimensionsBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "dimensions": ("DIMENSIONS", {}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0, "step": 0.001}),
        }}

    @staticmethod
    def f(dimensions: T.Dimensions, megapixels: float ) -> tuple[T.Dimensions]:
        width, height = dimensions
        curr_megapixels = (width * height) / 1_000_000
        scale = sqrt(megapixels / curr_megapixels)
        dimensions = (int(width * scale), int(height * scale))
        return (dimensions, )


class Change(DimensionsBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "dimensions": ("DIMENSIONS", {}),
            "delta_width": ("INT", {"default": 0}),
            "delta_height": ("INT", {"default": 0}),
        }}

    @staticmethod
    def f(dimensions: T.Dimensions, delta_width: int, delta_height: int) -> tuple[T.Dimensions]:
        width, height = dimensions
        new_width = width + delta_width
        new_height = height + delta_height
        return ((new_width, new_height), )


class AlignTo(DimensionsBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "dimensions": ("DIMENSIONS", {}),
            "align_to": ("INT", {"default": 8, "min": 1}),
        }}

    @staticmethod
    def f(dimensions: T.Dimensions, align_to: int) -> tuple[T.Dimensions]:
        width, height = dimensions
        new_width = U.round_up_to_mult_of(width, align_to)
        new_height = U.round_up_to_mult_of(height, align_to)
        return ((new_width, new_height), )


class FromRaw(DimensionsBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "width": ("INT", {"default": 1024, "min": 1}),
            "height": ("INT", {"default": 1024, "min": 1}),
        }}

    @staticmethod
    def f(width: int, height: int) -> tuple[T.Dimensions]:
        return ((width, height), )


class ToRaw:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"dimensions": ("DIMENSIONS", {})}}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")

    @staticmethod
    def f(dimensions: T.Dimensions) -> tuple[int, int]:
        width, height = dimensions
        return (width, height)


NODE_CLASS_MAPPINGS = {
    "Dimensions | Scale":               Scale,
    "Dimensions | Change":              Change,
    "Dimensions | Scale to Megapixels": ScaleToMegapixels,
    "Dimensions | Align To":            AlignTo,
    "Dimensions | From Raw":            FromRaw,
    "Dimensions | To Raw":              ToRaw,
    "Dimensions | From GTF":            FromGTF,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
