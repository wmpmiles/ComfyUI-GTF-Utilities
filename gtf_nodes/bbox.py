import torch
from .. import types as T
from ..gtf_impl import bbox as BB


class BBoxBase:
    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = 'gtf/bounding_box'
    FUNCTION = "f"


class FromMask(BBoxBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"mask": ("MASK", )}}

    @staticmethod
    def f(mask: torch.Tensor) -> tuple[T.BoundingBox]:
        mask_bbox = BB.from_mask(mask)
        return (mask_bbox, )


class Change(BBoxBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox":        ("BOUNDING_BOX", ),
            "delta_left":  ("INT", {"default": 0}),
            "delta_right": ("INT", {"default": 0}),
            "delta_up":    ("INT", {"default": 0}),
            "delta_down":  ("INT", {"default": 0}),
        }}

    @staticmethod
    def f(bbox: torch.Tensor, delta_left: int, delta_right: int, delta_up: int, delta_down: int) -> tuple[T.BoundingBox]:
        deltas = (delta_left, delta_right, delta_up, delta_down)
        changed_bbox = BB.pad(bbox, deltas)
        return (changed_bbox, )


class AreaScale(BBoxBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox":       ("BOUNDING_BOX", {}),
            "area_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
            "square":     ("BOOLEAN", {"default": True}),
        }}

    @staticmethod
    def f(bbox: T.BoundingBox, area_scale: float, square: bool) -> tuple[T.BoundingBox]:
        wh, lrud = bbox
        if square:
            scaled = BB.expand_lrud_square(lrud, wh, area_scale)
        else:
            scaled = BB.expand_lrud(lrud, wh, area_scale)
        return ((wh, scaled), )


NODE_CLASS_MAPPINGS = {
    "BBOX | From Mask":  FromMask,
    "BBOX | Change":     Change,
    "BBOX | Scale Area": AreaScale,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
