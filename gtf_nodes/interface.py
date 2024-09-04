from copy import copy
import torch
from .. import types as T


class InterfaceBase:
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"gtf": ("GTF", {})}}

    RETURN_TYPES = ('GTF', )
    RETURN_NAMES = ('gtf', )
    CATEGORY = 'gtf/interface'
    FUNCTION = "f"


class FromImage(InterfaceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"image": ("IMAGE", {})}}

    @staticmethod
    def f(image: torch.Tensor) -> tuple[torch.Tensor]:
        tensor = image.permute(0, 3, 1, 2)
        return (tensor, )


class FromMask(InterfaceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"mask": ("MASK", {})}}

    @staticmethod
    def f(mask: torch.Tensor) -> tuple[torch.Tensor]:
        tensor = mask.unsqueeze(1)
        return (tensor, )


class FromLatent(InterfaceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {"latent": ("LATENT", {})}}

    @staticmethod
    def f(latent: T.Latent) -> tuple[torch.Tensor]:
        tensor = latent["samples"]
        return (tensor, )


class ToImage(InterfaceBase):
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf.shape[1] not in (1, 3, 4):
            raise ValueError("Can only convert a 1, 3 or 4 channel GTF to an image.")
        if gtf.shape[1] == 1:
            gtf = gtf.expand(-1, 3, -1, -1)
        image = gtf.permute(0, 2, 3, 1)
        return (image, )


class ToMask(InterfaceBase):
    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )

    @staticmethod
    def f(gtf: torch.Tensor) -> tuple[torch.Tensor]:
        if gtf.shape[1] != 1:
            raise ValueError("Cannot convert multi-channel GTF to mask.")
        mask = gtf.squeeze(1)
        return (mask, )


class UpdateLatent(InterfaceBase):
    @staticmethod
    def INPUT_TYPES():
        return {"required": {
            "gtf": ("GTF", {}),
            "latent": ("LATENT", {}),
        }}

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("latent", )

    @staticmethod
    def f(gtf: torch.Tensor, latent: T.Latent) -> tuple[T.Latent]:
        channels = gtf.shape[1]
        expected_channels = latent["samples"].shape[1]
        if channels != expected_channels:
            raise ValueError(f"Expected GTF with {expected_channels} channels but GTF had {channels} channels.")
        updated_latent = copy(latent)
        updated_latent["samples"] = gtf
        return (updated_latent, )


NODE_CLASS_MAPPINGS = {
    "GTF | From Image":    FromImage,
    "GTF | From Mask":     FromMask,
    "GTF | From Latent":   FromLatent,
    "GTF | To Image":      ToImage,
    "GTF | To Mask":       ToMask,
    "GTF | Update Latent": UpdateLatent,
}

__all__ = ["NODE_CLASS_MAPPINGS"]
