import torch


class TrigBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {})}}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math/trigonometry"
    FUNCTION = "f"


class Sin(TrigBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        sin = torch.sin(gtf)
        return (sin, )


class Cos(TrigBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        cos = torch.cos(gtf)
        return (cos, )


class Tan(TrigBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        tan = torch.tan(gtf)
        return (tan, )


class Asin(TrigBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        asin = torch.asin(gtf)
        return (asin, )


class Acos(TrigBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        acos = torch.acos(gtf)
        return (acos, )


class Atan(TrigBase):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        atan = torch.atan(gtf)
        return (atan, )


class Atan2(TrigBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_x": ("GTF", {}),
            "gtf_y": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf_x: torch.Tensor, gtf_y: torch.Tensor) -> tuple[torch.Tensor]:
        atan2 = torch.atan2(gtf_y, gtf_x).nan_to_num()
        return (atan2, )


class Hypot(TrigBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_x": ("GTF", {}),
            "gtf_y": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf_x: torch.Tensor, gtf_y: torch.Tensor) -> tuple[torch.Tensor]:
        hypot = torch.hypot(gtf_x, gtf_y)
        return (hypot, )


NODE_CLASS_MAPPINGS = {
    "GTF | Trig - Sin":   Sin,
    "GTF | Trig - Cos":   Cos,
    "GTF | Trig - Tan":   Tan,
    "GTF | Trig - Asin":  Asin,
    "GTF | Trig - Acos":  Acos,
    "GTF | Trig - Atan":  Atan,
    "GTF | Trig - Atan2": Atan2,
    "GTF | Trig - Hypot": Hypot,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
