import torch
from typing import Any
from .transform import crop_uncrop, uncrop_bbox

class CropUncropRelative:
    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "dimensions": ("DIM", ),
                "anchor": (cls.ANCHORS, {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    ANCHORS = ("top-left", "top", "top-right", "left", "middle", "right", \
               "bottom-left", "bottom", "bottom-right")
    SINGLE_ANCHORS = ("left", "middle", "right")

    @classmethod
    def f(
        cls, 
        gtf: torch.Tensor, 
        dimensions: tuple[int, int], 
        anchor: str
    ) -> tuple[torch.Tensor]:
        width, height = dimensions
        index = cls.ANCHORS.index(anchor)
        width_anchor = cls.SINGLE_ANCHORS[index % 3]
        height_anchor = cls.SINGLE_ANCHORS[index // 3]
        cuc_width = crop_uncrop(gtf, 3, width, width_anchor)
        cuc_height = crop_uncrop(cuc_width, 2, height, height_anchor)
        return (cuc_height, )


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
    def f(
        gtf: torch.Tensor, 
        bbox: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[list[torch.Tensor]]:
        wh, lrud = bbox
        w, h = (int(x) for x in wh)
        if gtf.shape[2] != h or gtf.shape[3] != w:
            raise ValueError("GTF dimensions do not match those expected by the bounding box.")
        if gtf.shape[0] != lrud.shape[0]:
            raise ValueError("bbox and tensor batch size must match")
        cropped = []
        unbatched = torch.split(gtf, 1)
        lruds = (x.squeeze() for x in lrud.split(1))
        for single_lrud, single_tensor in zip(lruds, unbatched):
            l, r, u, d = (int(x) for x in single_lrud)
            cropped += [single_tensor[:, :, u:d, l:r]]
        return (cropped, )


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
            uncropped = uncrop_bbox(tensor, single_lrud, wh)
            uncropped_list += [(uncropped, typeinfo, extra)]
        return (uncropped_list, )


class BatchGTF:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf_1": ("GTF", {}),
                "gtf_2": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf_1: tuple[torch.Tensor, str, Any], gtf_2: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor1, typeinfo, extra = gtf_1
        tensor2, _, _ = gtf_2
        if tensor1.shape[1:] != tensor2.shape[1:]:
            raise ValueError("GTFs must have the same dimensions in all but batch count to be batched together.")
        tensor = torch.cat((tensor1, tensor2))
        return ((tensor, typeinfo, extra), )
