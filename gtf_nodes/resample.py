import torch
from .. import types as T
from ..gtf_impl import resample as RS


# BASE CLASSES

class ResampleBase:
    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"


# NODES

class ResampleCoords(ResampleBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "old_dimensions": ("DIMENSIONS", {}),
            "new_dimensions": ("DIMENSIONS", {}),
            "radius": ("INT", {"min": 0, "default": 1}),
        }}

    RETURN_TYPES = ("GTF", "GTF")
    RETURN_NAMES = ("gtf_x", "gtf_y")

    @staticmethod
    def f(old_dimensions: T.Dimensions, new_dimensions: T.Dimensions, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
        old_width, old_height = old_dimensions
        new_width, new_height = new_dimensions
        xs = RS.x_values_1d(old_width, new_width, radius)
        ys = RS.x_values_1d(old_height, new_height, radius)
        xl, xw = xs.shape
        yl, yw = ys.shape
        coords = (xs.reshape(1, 1, 1, xl, 1, xw), ys.reshape(1, 1, yl, 1, yw, 1))
        return (*coords, )


class ResampleSeparable(ResampleBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_filter_x": ("GTF", {}),
            "gtf_filter_y": ("GTF", {}),
            "radius": ("INT", {"min": 0, "default": 1}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_filter_x: torch.Tensor, gtf_filter_y: torch.Tensor, radius: int) -> tuple[torch.Tensor]:
        _, _, _, xl, _, xw = gtf_filter_x.shape
        _, _, yl, _, yw, _ = gtf_filter_y.shape
        resampled_x = RS.filter_1d(gtf, gtf_filter_x.reshape(xl, xw), xl, radius, 3)
        resampled_xy = RS.filter_1d(resampled_x, gtf_filter_y.reshape(yl, yw), yl, radius, 2)
        return (resampled_xy, )


class Resample2D(ResampleBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_filter": ("GTF", {}),
            "radius": ("INT", {"min": 0, "default": 1}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_filter: torch.Tensor, radius: int) -> tuple[torch.Tensor]:
        _, _, yl, xl, _, _ = gtf_filter.shape
        resampled = RS.filter_2d(gtf, gtf_filter.squeeze(), (yl, xl), radius, (2, 3))
        return (resampled, )


NODE_CLASS_MAPPINGS = {
    "GTF | Resample - Coords": ResampleCoords,
    "GTF | Resample - Separable": ResampleSeparable,
    "GTF | Resample - 2D": Resample2D,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
