import torch
from ..gtf_impl import utils as U
from ..gtf_impl import convert as CV


class ConvertBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {'gtf': ('GTF', {})}}

    RETURN_TYPES = ('GTF', )
    RETURN_NAMES = ('gtf', )
    CATEGORY = 'gtf/convert'
    FUNCTION = "f"


class Luminance(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        luminance = CV.to_luminance(gtf)
        return (luminance, )


class Min(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = U.gtf_min(gtf, (2, 3))
        return (tensor_min, )


class Max(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = U.gtf_max(gtf, (2, 3))
        return (tensor_max, )


class BatchMin(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = U.gtf_min(gtf, (0, ))
        return (tensor_min, )


class BatchMax(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = U.gtf_max(gtf, (0, ))
        return (tensor_max, )


class ChannelMin(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = U.gtf_min(gtf, (1, ))
        return (tensor_min, )


class ChannelMax(ConvertBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = U.gtf_max(gtf, (1, ))
        return (tensor_max, )


NODE_CLASS_MAPPINGS = {
    "GTF | Convert - Luminance":   Luminance,
    "GTF | Convert - Min":         Min,
    "GTF | Convert - Max":         Max,
    "GTF | Convert - Batch Min":   BatchMin,
    "GTF | Convert - Batch Max":   BatchMax,
    "GTF | Convert - Channel Min": ChannelMin,
    "GTF | Convert - Channel Max": ChannelMax,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
