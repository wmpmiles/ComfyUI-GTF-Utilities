import torch
from gtf_impl import bbox as bb


class FromMask:
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
        mask_bbox = bb.from_mask(mask)
        return (mask_bbox, )


class Change:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
            "delta_left": ("INT", {"default": 0}),
            "delta_right": ("INT", {"default": 0}),
            "delta_up": ("INT", {"default": 0}),
            "delta_down": ("INT", {"default": 0}),
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
        changed_bbox = bb.pad(bbox, deltas)
        return (changed_bbox, )


class AreaScale:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
            "area_scale": ("FLOAT", {"default": 1.0, "min": 0.0}),
            "square": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "gtf/bbox"
    FUNCTION = "f"

    @staticmethod
    def f(
        bbox: tuple[torch.Tensor, torch.Tensor],
        area_scale: float,
        square: bool
    ) -> tuple[tuple[torch.Tensor, torch.Tensor]]:
        wh, lrud = bbox
        if square:
            scaled = bb.expand_lrud_square(lrud, wh, area_scale)
        else:
            scaled = bb.expand_lrud(lrud, wh, area_scale)
        return ((wh, scaled), )
