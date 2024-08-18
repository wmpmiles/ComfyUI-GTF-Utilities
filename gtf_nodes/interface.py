from copy import copy
import torch


class FromImages:
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


class FromMasks:
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


class FromLatents:
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


class ToImages:
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
        if gtf.shape[1] not in (3, 4):
            raise ValueError("Can only convert 3 and 4 channel GTFs to \
                images.")
        images = gtf.permute(0, 2, 3, 1)
        return (images, )


class ToMasks:
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


class UpdateLatents:
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
        channels = gtf.shape[1]
        expected_channels = latents["samples"].shape[1]
        if channels != expected_channels:
            raise ValueError(f"Expected GTF with {expected_channels} channels, \
                but GTF had {channels} channels.")
        updated_latents = copy(latents)
        updated_latents["samples"] = gtf
        return (updated_latents, )
