import torch
from typing import Any
import torch.nn.functional as F
from .transform import crop_uncrop

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

    ANCHORS = ("top-left", "top", "top-right", "left", "middle", "right", "bottom-left", "bottom", "bottom-right")
    SINGLE_ANCHORS = ("left", "middle", "right")

    @classmethod
    def f(cls, gtf: tuple[torch.Tensor, str, Any], dimensions: tuple[int, int], anchor: str) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        width, height = dimensions
        tensor = tensor.clone()
        index = cls.ANCHORS.index(anchor)
        width_anchor = cls.SINGLE_ANCHORS[index % 3]
        height_anchor = cls.SINGLE_ANCHORS[index // 3]
        cropped_width = crop_uncrop(tensor, 3, width, width_anchor)
        cropped_height = crop_uncrop(cropped_width, 2, height, height_anchor)
        return ((cropped_height, typeinfo, extra), )

