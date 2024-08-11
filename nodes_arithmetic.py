import torch
from typing import Any


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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_a: tuple[torch.Tensor, str, Any], gtf_b: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor_a, typeinfo, extra = gtf_a
        tensor_b, _, _ = gtf_b
        added = tensor_a + tensor_b
        return ((added, typeinfo, extra), )


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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_a: tuple[torch.Tensor, str, Any], gtf_b: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor_a, typeinfo, extra = gtf_a
        tensor_b, _, _ = gtf_b
        multiplied = tensor_a * tensor_b
        return ((multiplied, typeinfo, extra), )


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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        reciprocal = torch.reciprocal(tensor)
        return ((reciprocal, typeinfo, extra), )


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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        negative = torch.negative(tensor)
        return ((negative, typeinfo, extra), )


class Zero:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {}
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f() -> tuple[tuple[torch.Tensor, str, Any]]:
        zero = torch.zeros(1, 1, 1, 1)
        return ((zero, "", None), )


class One:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {}
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f() -> tuple[tuple[torch.Tensor, str, Any]]:
        one = torch.ones(1, 1, 1, 1)
        return ((one, "", None), )


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
    CATEGORY = "gtf/arithmetic"
    FUNCTION = "f"

    @staticmethod
    def f(value: float) -> tuple[tuple[torch.Tensor, str, Any]]:
        one = torch.ones(1, 1, 1, 1) * value
        return ((one, "", None), )
