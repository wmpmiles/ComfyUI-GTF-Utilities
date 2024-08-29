import torch
from ..gtf_impl import window as WI


# BASE CLASSES

class WindowBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "radius": ("INT", {"min": 0, "default": 1}),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math/window"
    FUNCTION = "f"


# NODES

class Boxcar(WindowBase):
    @staticmethod
    def f(gtf: torch.Tensor, radius: int) -> tuple[torch.Tensor]:
        windowed = WI.boxcar(gtf, radius)
        return (windowed, )


class Triangle(WindowBase):
    @staticmethod
    def f(gtf: torch.Tensor, radius: int) -> tuple[torch.Tensor]:
        windowed = WI.triangle(gtf, radius)
        return (windowed, )


class Lanczos(WindowBase):
    @staticmethod
    def f(gtf: torch.Tensor, radius: int) -> tuple[torch.Tensor]:
        windowed = WI.lanczos(gtf, radius)
        return (windowed, )


class MitchellNetravali(WindowBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "b": ("FLOAT", {"default": 0.33, "min": 0, "max": 1, "step": 0.001}),
            "c": ("FLOAT", {"default": 0.33, "min": 0, "max": 1, "step": 0.001}),
        }}

    @staticmethod
    def f(gtf: torch.Tensor, b: float, c: float) -> tuple[torch.Tensor]:
        windowed = WI.mitchell_netravali(gtf, b, c)
        return (windowed, )


class Area(WindowBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "old_dim": ("INT", {"default": 1024, "min": 1}),
            "new_dim": ("INT", {"default": 1024, "min": 1}),
        }}
        
    @staticmethod
    def f(gtf: torch.Tensor, old_dim: int, new_dim: int) -> tuple[torch.Tensor]:
        windowed = WI.area(gtf, old_dim, new_dim)
        return (windowed, )


class MitchellNetravaliRadius(WindowBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {}}

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("radius", )

    @staticmethod
    def f() -> tuple[int]:
        radius = WI.mitchell_netravali_radius()
        return (radius, )


class AreaRadius(WindowBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {}}

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("radius", )

    @staticmethod
    def f() -> tuple[int]:
        radius = WI.area_window_radius()
        return (radius, )


NODE_CLASS_MAPPINGS = {
    "GTF | Window - Boxcar": Boxcar,
    "GTF | Window - Triangle": Triangle,
    "GTF | Window - Lanczos": Lanczos,
    "GTF | Window - Mitchell-Netravali": MitchellNetravali,
    "GTF | Window - Area": Area,
    "GTF | Helper - Mitchell Netravali Radius": MitchellNetravaliRadius,
    "GTF | Helper - Area Window Radius": AreaRadius,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
