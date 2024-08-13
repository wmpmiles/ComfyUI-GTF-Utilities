import torch
from .bbox import bounding_box_from_mask, pad_bounding_box, expand_lrud, expand_lrud_square


class MaskToBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "mask": ("MASK", ),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "gtf/bbox"
    FUNCTION = "f"

    @staticmethod
    def f(mask: torch.Tensor) -> tuple[torch.Tensor]:
        bbox = bounding_box_from_mask(mask)
        return (bbox, )


class ChangeBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
            "delta_left": ("INT", { "default": 0 }),
            "delta_right": ("INT", { "default": 0 }),
            "delta_up": ("INT", { "default": 0 }),
            "delta_down": ("INT", { "default": 0 }),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "gtf/bbox"
    FUNCTION = "f"

    @staticmethod
    def f(
        bbox: torch.Tensor, 
        delta_left: int, 
        delta_right: int, 
        delta_up: int, 
        delta_down: int
    ) -> tuple[torch.Tensor]:
        deltas = (delta_left, delta_right, delta_up, delta_down)
        changed_bbox = pad_bounding_box(bbox, deltas)
        return (changed_bbox, )


class BoundingBoxAreaScale:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
            "area_scale": ("FLOAT", { "default": 1.0, "min": 0.0, }),
            "square": ("BOOLEAN", { "default": True }),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "gtf/bbox"
    FUNCTION = "f"

    @staticmethod
    def f(bbox: tuple[torch.Tensor, torch.Tensor], area_scale: float, square: bool) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        wh, lrud = bbox
        if square:
            scaled = expand_lrud_square(lrud, wh, area_scale)
        else:
            scaled = expand_lrud(lrud, wh, area_scale)
        return ((wh, scaled), )

