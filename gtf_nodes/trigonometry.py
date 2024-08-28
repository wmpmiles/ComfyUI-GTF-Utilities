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
            "gtf_opposite": ("GTF", {}),
            "gtf_adjacent": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf_opposite: torch.Tensor, gtf_adjacent: torch.Tensor) -> tuple[torch.Tensor]:
        atan2 = torch.atan2(gtf_opposite, gtf_adjacent)
        return (atan2, )


NODE_CLASS_MAPPINGS = {
    "GTF | Trig - Sin":   Sin,
    "GTF | Trig - Cos":   Cos,
    "GTF | Trig - Tan":   Tan,
    "GTF | Trig - Asin":  Asin,
    "GTF | Trig - Acos":  Acos,
    "GTF | Trig - Atan":  Atan,
    "GTF | Trig - Atan2": Atan2,
}