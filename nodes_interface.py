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
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(images: torch.Tensor) -> tuple[torch.Tensor]:
        tensor = images.permute(0, 3, 1, 2)
        return (tensor, )


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
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(masks: torch.Tensor) -> tuple[torch.Tensor]:
        tensor = masks.unsqueeze(1)
        return (tensor, )


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
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(latents: dict[str, torch.Tensor]) -> tuple[torch.Tensor]:
        tensor = latents["samples"]
        return (tensor, )


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
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        images = gtf.permute(0, 2, 3, 1)
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
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf.shape[1] != 1:
            raise ValueError("Cannot convert multi-channel GTF to mask.")
        masks = gtf.squeeze(1)
        return (masks, )


class GTFToNewLatents:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
            },
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latents", )
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[dict]:
        latents = {"samples": gtf}
        return (latents, )


class GTFUpdateLatents:
    @staticmethod
    def INPUT_TYPES():
        return { 
            "required": {
                "gtf": ("GTF", {}),
                "latents": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latents", )
    CATEGORY = "gtf/interface"
    FUNCTION = "f"

    @staticmethod
    def f(gtf: torch.Tensor, latents: dict[str, torch.Tensor]) -> tuple[dict]:
        updated_latents = copy(latents)
        updated_latents["samples"] = gtf
        return (latents, )
