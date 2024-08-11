import torch
from typing import Any
from .bbox import bounding_box_from_mask, pad_bounding_box, expand_lrud, expand_lrud_square, uncrop


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


class CropToBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    OUTPUT_IS_LIST = (True, )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], bbox: torch.Tensor) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        wh, lrud = bbox
        w, h = (int(x) for x in wh)
        if tensor.shape[2] != h or tensor.shape[3] != w:
            raise ValueError("GTF dimensions do not match those expected by the bounding box.")
        if tensor.shape[0] != lrud.shape[0]:
            raise ValueError("bbox and tensor batch size must match")
        cropped = []
        unbatched = torch.split(tensor, 1)
        lruds = (x.squeeze() for x in lrud.split(1))
        for single_lrud, single_tensor in zip(lruds, unbatched):
            l, r, u, d = (int(x) for x in single_lrud)
            cropped += [single_tensor[:, :, u:d, l:r]]
        ret = [(x, typeinfo, extra) for x in cropped]
        return (ret, )


class UncropFromBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    OUTPUT_IS_LIST = (True, )
    INPUT_IS_LIST = (True, False)
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: list[tuple[torch.Tensor, str, Any]], bbox: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[tuple[torch.Tensor, str, Any]]:
        wh, lrud = bbox[0]
        if lrud.shape[0] != len(gtf):
            raise ValueError("GTF and bbox batch size must match.")
        unbatched = (x.squeeze() for x in lrud.split(1))
        uncropped_list = []
        for single_gtf, single_lrud in zip(gtf, unbatched):
            tensor, typeinfo, extra = single_gtf
            uncropped = uncrop(tensor, single_lrud, wh)
            uncropped_list += [(uncropped, typeinfo, extra)]
        return (uncropped_list, )

