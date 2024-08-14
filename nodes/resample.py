import torch
from ..impl.resample import nearest_neighbor_resample_2d, filter_resample_2d, \
    filter_resample_2d_seperable, area_resample_2d, triangle_filter, \
    lanczos_filter, mitchell_netravali_filter, mitchell_netravali_radius

class ResampleNearestNeighbor:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        dimensions: tuple[int, int],
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        resampled = nearest_neighbor_resample_2d(gtf, (height, width), (2, 3))
        return (resampled, )


class ResampleTriangle:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "radius": ("INT", { "default": 1, "min": 1 }),
                "seperable": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        dimensions: tuple[int, int], 
        radius: int, 
        seperable: bool,
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        filter = triangle_filter(radius)
        fn = filter_resample_2d_seperable if seperable else filter_resample_2d
        resampled = fn(gtf, (height, width), radius, filter, (2, 3))
        return (resampled, )


class ResampleLanczos:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "radius": ("INT", { "default": 4, "min": 1 }),
                "seperable": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        dimensions: tuple[int, int], 
        radius: int, 
        seperable: bool,
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        filter = lanczos_filter(radius)
        fn = filter_resample_2d_seperable if seperable else filter_resample_2d
        resampled = fn(gtf, (height, width), radius, filter, (2, 3))
        return (resampled, )


class ResampleMitchellNetravali:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "b": ("FLOAT", 
                        { "default": 0.33, "min": 0, "max": 1, "step": 0.01 }),
                "c": ("FLOAT", 
                        { "default": 0.33, "min": 0, "max": 1, "step": 0.01 }),
                "seperable": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        dimensions: tuple[int, int], 
        b: float, 
        c: float, 
        seperable: bool
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        filter = mitchell_netravali_filter(b, c)
        fn = filter_resample_2d_seperable if seperable else filter_resample_2d
        radius = mitchell_netravali_radius()
        resampled = fn(gtf, (height, width), radius, filter, (2, 3))
        return (resampled, )


class ResampleArea:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/resample"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf: torch.Tensor, 
        dimensions: tuple[int, int]
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        resampled = area_resample_2d(gtf, (height, width), (2, 3))
        return (resampled, )
