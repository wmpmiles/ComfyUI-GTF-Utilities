import torch
from typing import Any
from .bbox import bounding_box_from_mask, pad_bounding_box, expand_bounding_box, clamp_bounding_box


class MaskToBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "mask": ("MASK", ),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "bbox"
    FUNCTION = "f"

    @staticmethod
    def f(mask: torch.Tensor) -> tuple[torch.Tensor]:
        if int(mask.shape[0]) != 1:
            raise ValueError("Cannot calculate bounding box for more than 1 mask at a time.")
        bbox = bounding_box_from_mask(mask).squeeze()
        return (bbox, )


class PadBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
            "pad_left": ("INT", { "default": 0, "min": 0, }),
            "pad_right": ("INT", { "default": 0, "min": 0, }),
            "pad_up": ("INT", { "default": 0, "min": 0, }),
            "pad_down": ("INT", { "default": 0, "min": 0, }),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "bbox"
    FUNCTION = "f"

    @staticmethod
    def f(bbox: torch.Tensor, pad_left: int, pad_right: int, pad_up: int, pad_down: int) -> tuple[torch.Tensor]:
        bbox = pad_bounding_box(bbox.unsqueeze(0), (pad_left, pad_right, pad_up, pad_down)).squeeze()
        return bbox


class BoundingBoxAreaScale:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
            "area_scale": ("FLOAT", { "default": 1.0, "min": 0.0, }),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "bbox"
    FUNCTION = "f"

    @staticmethod
    def f(bbox: torch.Tensor, area_scale: float) -> tuple[torch.Tensor]:
        bbox = expand_bounding_box(bbox.unsqueeze(0), area_scale).squeeze()
        return (bbox, )


class BoundingBoxToNearest8:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("BOUNDING_BOX", )
    RETURN_NAMES = ("bbox", )
    CATEGORY = "bbox"
    FUNCTION = "f"

    @staticmethod
    def f(bbox: torch.Tensor) -> tuple[torch.Tensor]:
        term = torch.tensor([7, 7, 7, 7, 0, 0])
        coef = torch.tensor([8, 8, 8, 8, 1, 1])
        ceil8 = ((bbox + term) // coef) * coef
        clamped = clamp_bounding_box(ceil8.unsqueeze(0)).squeeze()
        return (clamped, )


class CropToBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], bbox: torch.Tensor) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        (l, r, u, d, w, h) = (int(x) for x in bbox)
        if tensor.shape[2] != h or tensor.shape[3] != w:
            raise ValueError("GTF dimensions do not match those expected by the bounding box.")
        cropped = tensor[:, :, u:d, l:r]
        return ((cropped, typeinfo, extra), )


class UncropFromBoundingBox:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", ),
            "bbox": ("BOUNDING_BOX", ),
        }}

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], bbox: torch.Tensor) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        b, c, th, tw = (int(x) for x in tensor.shape)
        l, r, u, d, w, h = (int(x) for x in bbox)
        bh, bw = (d - u, r - l)
        if th != bh or tw != bw:
            raise ValueError("GTF dimensions do not match bounding box dimensions.")
        uncropped = torch.zeros(b, c, h, w)
        uncropped[:,:,u:d,l:r] = tensor
        return ((uncropped, typeinfo, extra), )

