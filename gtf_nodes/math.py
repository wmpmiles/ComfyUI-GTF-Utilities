import torch


class Add:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_a": ("GTF", ),
                "gtf_b": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_a: torch.Tensor, gtf_b: torch.Tensor) -> tuple[torch.Tensor]:
        added = gtf_a + gtf_b
        return (added, )


class Multiply:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_a": ("GTF", ),
                "gtf_b": ("GTF", ),
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_a: torch.Tensor, gtf_b: torch.Tensor) -> tuple[torch.Tensor]:
        multiplied = gtf_a * gtf_b
        return (multiplied, )


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


class Zero:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {}
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f() -> tuple[torch.Tensor]:
        zero = torch.zeros(1, 1, 1, 1)
        return (zero, )


class One:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {}
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f() -> tuple[torch.Tensor]:
        one = torch.ones(1, 1, 1, 1)
        return (one, )


class Float:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0})
            }
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/math"
    FUNCTION = "f"

    @staticmethod
    def f(value: float) -> tuple[torch.Tensor]:
        value_tensor = torch.tensor(value).reshape(1, 1, 1, 1)
        return (value_tensor, )


class Lerp:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "gtf_0": ("GTF", ),
                "gtf_1": ("GTF", ),
                "t": ("FLOAT", {"default": 0.5})
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
        lerped = torch.lerp(gtf0, gtf_1, t)
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
