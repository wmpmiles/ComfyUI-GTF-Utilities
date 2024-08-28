import torch
from ..gtf_impl import resample as R


def select_fn(seperable: bool):
    if seperable:
        return R.filter_2d_seperable
    else:
        return R.filter_2d


class NearestNeighbor:
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
        resampled = \
            R.nearest_neighbor_2d(gtf, (height, width), (2, 3))
        return (resampled, )


class Triangle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "radius": ("INT", {"default": 1, "min": 1}),
                "seperable": ("BOOLEAN", {"default": True}),
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
        filter = R.triangle_filter(radius)
        fn = select_fn(seperable)

        def function(x):
            return fn(gtf(x), (height, width), radius, filter, (2, 3))

        return (function, )


class Lanczos:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "radius": ("INT", {"default": 4, "min": 1}),
                "seperable": ("BOOLEAN", {"default": True}),
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
        filter = R.lanczos_filter(radius)
        fn = select_fn(seperable)
        resampled = fn(gtf, (height, width), radius, filter, (2, 3))
        return (resampled, )


class MitchellNetravali:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", {}),
                "b": ("FLOAT",
                      {"default": 0.33, "min": 0, "max": 1, "step": 0.01}),
                "c": ("FLOAT",
                      {"default": 0.33, "min": 0, "max": 1, "step": 0.01}),
                "seperable": ("BOOLEAN", {"default": True}),
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
        filter = R.mitchell_netravali_filter(b, c)
        fn = select_fn(seperable)
        radius = R.mitchell_netravali_radius()
        resampled = fn(gtf, (height, width), radius, filter, (2, 3))
        return (resampled, )


class Area:
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
        resampled = R.area_2d(gtf, (height, width), (2, 3))
        return (resampled, )
