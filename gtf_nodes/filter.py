import torch
from gtf_impl import filter as F


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


class BlurGaussian:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
                "sigma": ("FLOAT", {"default": 3.0, "min": 0.0, "step": 0.1}),
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
        blurred = F.blur_gaussian(gtf, sigma)
        return (blurred, )
