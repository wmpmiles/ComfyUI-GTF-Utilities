import torch
from ..gtf_impl import utils as U
from ..gtf_impl import filters as FT


class FilterBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {'gtf': ('GTF', {})}}

    RETURN_TYPES = ('GTF', )
    RETURN_NAMES = ('gtf', )
    CATEGORY = 'gtf/filter'
    FUNCTION = "f"


class Invert(FilterBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        inverted = U.invert(gtf)
        return (inverted, )


class SumNormalize(FilterBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        normalized = U.sum_normalize(gtf, (2, 3))
        return (normalized, )


class RangeNormalize(FilterBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        normalized = U.range_normalize(gtf, (2, 3))
        return (normalized, )


class BinaryThreshold(FilterBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            'gtf': ('GTF', ),
            "threshold": ("FLOAT", {"default": 0.5, "step": 0.001}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, threshold: float) -> tuple[torch.Tensor]:
        thresholded = (gtf >= threshold).to(torch.float)
        return (thresholded, )


class Quantize(FilterBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            'gtf': ('GTF', ),
            "steps": ("INT", {"default": 256, "min": 2, "max": 1_000_000}),
            "mode": ([*FT.F_MAP.keys()], ),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, steps: int, mode: str) -> tuple[torch.Tensor]:
        quantized = FT.quantize(gtf, steps, mode)
        return (quantized, )


class Convolve(FilterBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "gtf_kernel": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, gtf_kernel: torch.Tensor) -> tuple[torch.Tensor]:
        convolved = FT.convolve_2d(gtf, gtf_kernel)
        return (convolved, )


class MorphologicalFilter:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "operation": (["dilate", "erode", "open", "close"], {}),
            "radius": ("INT", {"default": 3, "min": 1})
        }}

    @staticmethod
    def f(gtf: torch.Tensor, operation: str, radius: int) -> tuple[torch.Tensor]:
        if radius == 0:
            return (gtf, )
        match operation:
            case "dilate":
                filtered = FT.dilate(gtf, radius)
            case "erode":
                filtered = FT.erode(gtf, radius)
            case "open":
                filtered = FT.open(gtf, radius)
            case "close":
                filtered = FT.close(gtf, radius)
        return (filtered, )


NODE_CLASS_MAPPINGS = {
    "GTF | Filter - Convolve":         Convolve,
    "GTF | Filter - Sum Normalize":    SumNormalize,
    "GTF | Filter - Range Normalize":  RangeNormalize,
    "GTF | Filter - Invert":           Invert,
    "GTF | Filter - Binary Threshold": BinaryThreshold,
    "GTF | Filter - Quantize":         Quantize,
    "GTF | Filter - Morphological":    MorphologicalFilter,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
