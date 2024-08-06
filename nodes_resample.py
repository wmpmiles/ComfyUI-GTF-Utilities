import torch
from typing import Any
from .resample import nearest_neighbor_resample_2d, filter_resample_2d, filter_resample_2d_seperable, area_resample_2d, triangle_filter, lanczos_filter, mitchell_netravali_filter, mitchell_netravali_radius

class ResampleNearestNeighbor:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "width": ("INT", { "default": 1024, "min": 1 }),
                "height": ("INT", { "default": 1024, "min": 1 }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], width: int, height: int) -> tuple[torch.Tensor, str, Any]:
        tensor, typeinfo, extra = gtf
        resampled = nearest_neighbor_resample_2d(tensor, (height, width), (2, 3))
        return ((resampled, typeinfo, extra), )


class ResampleTriangle:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "width": ("INT", { "default": 1024, "min": 1 }),
                "height": ("INT", { "default": 1024, "min": 1 }),
                "radius": ("INT", { "default": 1, "min": 1 }),
                "seperable": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], width: int, height: int, radius: int, seperable: bool) -> tuple[torch.Tensor, str, Any]:
        tensor, typeinfo, extra = gtf
        filter = triangle_filter(radius)
        fn = filter_resample_2d_seperable if seperable else filter_resample_2d
        resampled = fn(tensor, (height, width), radius, filter, (2, 3))
        return ((resampled, typeinfo, extra), )


class ResampleLanczos:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "width": ("INT", { "default": 1024, "min": 1 }),
                "height": ("INT", { "default": 1024, "min": 1 }),
                "radius": ("INT", { "default": 3, "min": 1 }),
                "seperable": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], width: int, height: int, radius: int, seperable: bool) -> tuple[torch.Tensor, str, Any]:
        tensor, typeinfo, extra = gtf
        filter = lanczos_filter(radius)
        fn = filter_resample_2d_seperable if seperable else filter_resample_2d
        resampled = fn(tensor, (height, width), radius, filter, (2, 3))
        return ((resampled, typeinfo, extra), )


class ResampleMitchellNetravali:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "width": ("INT", { "default": 1024, "min": 1 }),
                "height": ("INT", { "default": 1024, "min": 1 }),
                "b": ("FLOAT", { "default": 0.33, "min": 0, "max": 1, "step": 0.01 }),
                "c": ("FLOAT", { "default": 0.33, "min": 0, "max": 1, "step": 0.01 }),
                "seperable": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], width: int, height: int, b: float, c: float, seperable: bool) -> tuple[torch.Tensor, str, Any]:
        tensor, typeinfo, extra = gtf
        filter = mitchell_netravali_filter(b, c)
        fn = filter_resample_2d_seperable if seperable else filter_resample_2d
        resampled = fn(tensor, (height, width), mitchell_netravali_radius(), filter, (2, 3))
        return ((resampled, typeinfo, extra), )


class ResampleArea:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "width": ("INT", { "default": 1024, "min": 1 }),
                "height": ("INT", { "default": 1024, "min": 1 }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], width: int, height: int) -> tuple[torch.Tensor, str, Any]:
        tensor, typeinfo, extra = gtf
        resampled = area_resample_2d(tensor, (height, width), (2, 3))
        return ((resampled, typeinfo, extra), )
