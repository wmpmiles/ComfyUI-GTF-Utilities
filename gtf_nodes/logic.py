import torch


# BASE CLASSES

class LogicBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf_lhs": ("GTF", ),
            "gtf_rhs": ("GTF", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/logic"
    FUNCTION = "f"


# NODES

class Equal(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        equal = (gtf_lhs == gtf_rhs)
        return (equal, )


class LessThan(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        less_than = (gtf_lhs < gtf_rhs)
        return (less_than, )


class LessThanOrEqual(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        less_than_or_equal = (gtf_lhs <= gtf_rhs)
        return (less_than_or_equal, )


class GreaterThan(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        greater_than = (gtf_lhs > gtf_rhs)
        return (greater_than, )


class GreaterThanOrEqual(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        greater_than_or_equal = (gtf_lhs >= gtf_rhs)
        return (greater_than_or_equal, )


class And(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        and_ = torch.logical_and(gtf_lhs, gtf_rhs)
        return (and_, )


class Or(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        or_ = torch.logical_or(gtf_lhs, gtf_rhs)
        return (or_, )


class Xor(LogicBase):
    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        xor_ = torch.logical_xor(gtf_lhs, gtf_rhs)
        return (xor_, )


class Not(LogicBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", )}}

    @staticmethod
    def f(gtf_lhs: torch.Tensor, gtf_rhs: torch.Tensor) -> tuple[torch.Tensor]:
        and_ = torch.logical_and(gtf_lhs, gtf_rhs)
        return (and_, )


NODE_CLASS_MAPPINGS = {
    "GTF | Logic - Equal":                 Equal,
    "GTF | Logic - Less Than":             LessThan,
    "GTF | Logic - Less Than or Equal":    LessThanOrEqual,
    "GTF | Logic - Greater Than":          GreaterThan,
    "GTF | Logic - Greater Than or Equal": GreaterThanOrEqual,
    "GTF | Logic - And":                   And,
    "GTF | Logic - Or":                    Or,
    "GTF | Logic - Xor":                   Xor,
    "GTF | Logic - Not":                   Not,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
