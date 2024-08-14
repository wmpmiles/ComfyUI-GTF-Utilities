import torch
from ..impl.convert import to_luminance
from ..impl.utils import gtf_min_max


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
        luminance = to_luminance(gtf)
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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = gtf_min_max(gtf, (2, 3), False)
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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = gtf_min_max(gtf, (2, 3), True)
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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = gtf_min_max(gtf, (0), False)
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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = gtf_min_max(gtf, (0), True)
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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_min = gtf_min_max(gtf, (0), False)
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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tensor_max = gtf_min_max(gtf, (0), True)
        return (tensor_max, )
