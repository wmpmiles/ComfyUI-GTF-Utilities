import torch
from typing import Any
import torch.nn.functional as F

class CropRelative:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "width": ("INT", { "default": 1024, "min": 0 }),
                "height": ("INT", { "default": 1024, "min": 0 }),
                "anchor": (("top-left", "top", "top-right", "left", "middle", "right", "bottom-left", "bottom", "bottom-right"), {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any], width: int, height: int, anchor: str) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        _, _, h, w = map(int, tensor.shape)
        height = min(height, h)
        width = min(width, w)
        if "left" in anchor:
            tensor = tensor[:,:,:,:width]
        elif "right" in anchor:
            tensor = tensor[:,:,:,w-width:]
        else:
            left = (w-width) // 2
            right = (w-width) - left
            tensor = tensor[:,:,:,left:-right]
        if "top" in anchor:
            tensor = tensor[:,:,:height,:]
        elif "bottom" in anchor:
            tensor = tensor[:,:,h-height:,:]
        else:
            top = (h-height) // 2
            bottom = (h-height) - top
            tensor = tensor[:,:,top:-bottom,:]
        return ((tensor, typeinfo, extra), )


class PadToMultOf8:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/transform"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor, typeinfo, extra = gtf
        _, _, h, w = map(int, tensor.shape)
        pad_b = (((h + 7) // 8) * 8) - h
        pad_r = (((w + 7) // 8) * 8) - w
        padded = F.pad(tensor, (0, pad_r, 0, pad_b), mode="reflect")
        return ((padded, typeinfo, extra), )
