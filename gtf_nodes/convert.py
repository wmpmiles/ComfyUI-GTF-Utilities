import torch
from gtf_impl import convert as C
from gtf_impl.utils import gtf_min, gtf_max


class Luminance:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        luminance = C.to_luminance(gtf)
        return (luminance, )


class Min:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = gtf_min(gtf, (2, 3))
        return (tensor_min, )


class Max:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = gtf_max(gtf, (2, 3))
        return (tensor_max, )


class BatchMin:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = gtf_min(gtf, (0))
        return (tensor_min, )


class BatchMax:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = gtf_max(gtf, (0))
        return (tensor_max, )


class ChannelMin:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = gtf_min(gtf, (0))
        return (tensor_min, )


class ChannelMax:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = gtf_max(gtf, (0))
        return (tensor_max, )


class BinaryThreshold:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
                "threshold": ("FLOAT", {"default": 0.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, threshold: float) -> tuple[torch.Tensor]:
        thresholded = (gtf >= threshold).to(torch.float)
        return (thresholded, )


class QuantizeNormalized:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
                "steps": ("INT", {"default": 256, "min": 2, "max": 1_000_000}),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/convert"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, steps: int) -> tuple[torch.Tensor]:
        quantized = C.quantize_normalized(gtf, steps)
        return (quantized, )
