from copy import copy
from typing import Any
import torch


class ImagesToGTF:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "images": ("IMAGE", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/conversion"
    FUNCTION = "f"

    @staticmethod
    def f(images: torch.Tensor) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor = images.permute(0, 3, 1, 2)
        return ((tensor, "IMAGE", None), )


class MasksToGTF:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "masks": ("MASK", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/conversion"
    FUNCTION = "f"

    @staticmethod
    def f(masks: torch.Tensor) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor = masks.unsqueeze(1)
        return ((tensor, "MASK", None), )


class LatentsToGTF:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "latents": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("GTF", )
    RETURN_NAMES = ("gtf", )
    CATEGORY = "gtf/conversion"
    FUNCTION = "f"

    @staticmethod
    def f(latents: dict[str, torch.Tensor]) -> tuple[tuple[torch.Tensor, str, Any]]:
        tensor = latents["samples"]
        return ((tensor, "LATENT", copy(latents)), )


class GTFToImages:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    CATEGORY = "gtf/conversion"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[torch.Tensor]:
        tensor, typeinfo, _ = gtf
        if typeinfo != "IMAGE":
            raise ValueError("Cannot convert GTF from non-image source back to image.")
        images = tensor.permute(0, 2, 3, 1)
        return (images, )


class GTFToMasks:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("masks", )
    CATEGORY = "gtf/conversion"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[torch.Tensor]:
        tensor, typeinfo, _ = gtf
        if typeinfo != "MASK":
            raise ValueError("Cannot convert GTF from non-mask source back to masks.")
        masks = tensor.squeeze(1)
        return (masks, )


class GTFToLatents:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latents", )
    CATEGORY = "gtf/conversion"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: tuple[torch.Tensor, str, Any]) -> tuple[dict]:
        tensor, typeinfo, latents = gtf
        if typeinfo != "LATENT":
            raise ValueError("Cannot convert GTF from non-latent source back to latents.")
        latents["samples"] = tensor
        return (latents, )
