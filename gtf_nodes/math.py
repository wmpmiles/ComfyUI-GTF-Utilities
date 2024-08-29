import torch


# BASE CLASSES

class MathBase:
    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"


class MathBase2To1(MathBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_lhs": ("GTF", {}),
            "gtf_rhs": ("GTF", {}),
        }}


class MathBase1To1(MathBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {})}}


# NODES

class Add(MathBase2To1):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        added = gtf_lhs + gtf_rhs
        return (added, )


class Subtract(MathBase2To1):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        subtracted = gtf_lhs - gtf_rhs
        return (subtracted, )


class Multiply(MathBase2To1):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        multiplied = gtf_lhs * gtf_rhs
        return (multiplied, )


class Divide(MathBase2To1):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        divided = gtf_lhs / gtf_rhs
        return (divided, )


class Reciprocal(MathBase1To1):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        reciprocal = torch.reciprocal(gtf)
        return (reciprocal, )


class Negate(MathBase1To1):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        negated = torch.negative(gtf)
        return (negated, )


class Absolute(MathBase1To1):
    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        absolute = torch.abs(gtf)
        return (absolute, )


class Lerp(MathBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_0": ("GTF", {}),
            "gtf_1": ("GTF", {}),
            "t": ("FLOAT", {"default": 0.5, "step": 0.01})
        }}

    @staticmethod
    def f(gtf_0: torch.Tensor, gtf_1: torch.Tensor, t: float) -> tuple[torch.Tensor]:
        lerped = torch.lerp(gtf_0, gtf_1, t)
        return (lerped, )


class CartesianToPolar(MathBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_x": ("GTF", {}),
            "gtf_y": ("GTF", {}),
        }}

    RETURN_TYPES = ("GTF", "GTF")
    RETURN_NAMES = ("gtf_r", "gtf_theta")

    @staticmethod
    def f(gtf_x: torch.Tensor, gtf_y: torch.Tensor) -> tuple[torch.Tensor]:
        r = torch.hypot(gtf_x, gtf_y)
        theta = torch.atan2(gtf_y, gtf_x).nan_to_num()

        return (r, theta)


class PolarToCartesian(MathBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_r": ("GTF", {}),
            "gtf_theta": ("GTF", {}),
        }}

    RETURN_TYPES = ("GTF", "GTF")
    RETURN_NAMES = ("gtf_x", "gtf_y")

    @staticmethod
    def f(gtf_r: torch.Tensor, gtf_theta: torch.Tensor) -> tuple[torch.Tensor]:
        x = gtf_r * torch.cos(gtf_theta)
        y = gtf_r * torch.sin(gtf_theta)
        return (x, y)


class Pow(MathBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_base": ("GTF", {}),
            "gtf_exp": ("GTF", {}),
        }}

    @staticmethod
    def f(gtf_base: torch.Tensor, gtf_exp: torch.Tensor) -> tuple[torch.Tensor]:
        powed = gtf_base.pow(gtf_exp)
        return (powed, )


NODE_CLASS_MAPPINGS = {
    "GTF | Math - Add":        Add,
    "GTF | Math - Subtract":   Subtract,
    "GTF | Math - Multiply":   Multiply,
    "GTF | Math - Divide":     Divide,
    "GTF | Math - Reciprocal": Reciprocal,
    "GTF | Math - Negative":   Negate,
    "GTF | Math - Lerp":       Lerp,
    "GTF | Math - Pow":        Pow,
    "GTF | Math - Absolute":   Absolute,
    "GTF | Math - Cartesian to Polar": CartesianToPolar,
    "GTF | Math - Polar to Cartesian": PolarToCartesian,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
