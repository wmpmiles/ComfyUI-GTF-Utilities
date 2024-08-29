import torch


# BASE CLASSES

class SourceBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/source"
    FUNCTION = "f"


class KernelBase(SourceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "radius": ("INT", {"min": 0, "default": 1})
        }}

    RETURN_TYPES = ("GTF", )
    # Define RETURN_NAMES and SHAPE to finish class

    @classmethod
    def f(cls, radius: int) -> tuple[torch.Tensor]:
        width = 2 * radius + 1
        indices = torch.arange(width)
        offset = indices - radius
        reshaped = offset.reshape(*cls.SHAPE)
        return (reshaped, )


# NODES

class Zero(SourceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        zero = torch.zeros_like(gtf)
        return (zero, )


class One(SourceBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        one = torch.ones_like(gtf)
        return (one, )


class Value(SourceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "value": ("FLOAT", {"default": 0.0, "step": 0.0001, "min": -1_000_000, "max": 1_000_000})
        }}

    @staticmethod
    def f(gtf: torch.Tensor, value: float) -> tuple[torch.Tensor]:
        values = torch.ones_like(gtf) * value
        return (values, )


class RGB(SourceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "r": ("INT", {"default": 0, "min": 0, "max": 255}),
            "g": ("INT", {"default": 0, "min": 0, "max": 255}),
            "b": ("INT", {"default": 0, "min": 0, "max": 255}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, r: int, g: int, b: int) -> tuple[torch.Tensor]:
        b, _, h, w = gtf.shape
        rgb = (torch.tensor((r, g, b)).to(torch.float) / 255).reshape(1, 3, 1, 1).expand(b, -1, h, w)
        return (rgb, )


class KernelX(KernelBase):
    RETURN_NAMES = ("gtf_kernel_x", )
    SHAPE = (1, 1, 1, -1)


class KernelY(KernelBase):
    RETURN_NAMES = ("gtf_kernel_y", )
    SHAPE = (1, 1, -1, 1)


# MAPPING

NODE_CLASS_MAPPINGS = {
    "GTF | Source - Zero":  Zero,
    "GTF | Source - One":   One,
    "GTF | Source - Value": Value,
    "GTF | Source - RGB":   RGB,
    "GTF | Source - Kernel X": KernelX,
    "GTF | Source - Kernel Y": KernelY,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
