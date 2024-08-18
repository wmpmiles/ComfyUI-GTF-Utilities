import torch


class Add:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_lhs": ("GTF", ),
                "gtf_rhs": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        added = gtf_lhs + gtf_rhs
        return (added, )


class Subtract:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_lhs": ("GTF", ),
                "gtf_rhs": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        subtracted = gtf_lhs - gtf_rhs
        return (subtracted, )


class Multiply:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_lhs": ("GTF", ),
                "gtf_rhs": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        multiplied = gtf_lhs * gtf_rhs
        return (multiplied, )


class Divide:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_lhs": ("GTF", ),
                "gtf_rhs": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        divided = gtf_lhs / gtf_rhs
        return (divided, )


class Reciprocal:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        reciprocal = torch.reciprocal(gtf)
        return (reciprocal, )


class Negate:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        negated = torch.negative(gtf)
        return (negated, )


class Lerp:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_0": ("GTF", ),
                "gtf_1": ("GTF", ),
                "t": ("FLOAT", {"default": 0.5, "step": 0.01})
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf_0: torch.Tensor,
        gtf_1: torch.Tensor,
        t: float
    ) -> tuple[torch.Tensor]:
        lerped = torch.lerp(gtf_0, gtf_1, t)
        return (lerped, )


class Pow:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_base": ("GTF", ),
                "gtf_exp": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(
        gtf_base: torch.Tensor,
        gtf_exp: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        powed = gtf_base.pow(gtf_exp)
        return (powed, )
