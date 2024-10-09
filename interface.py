import torch
from torch import Tensor


def gtf_from_image(image: Tensor) -> Tensor:
    gtf = image.permute(0, 3, 1, 2)
    return gtf


def gtf_from_mask(mask: Tensor) -> Tensor:
    gtf = mask.unsqueeze(1)
    return gtf


def gtf_from_latent(latent: dict[str, Tensor]) -> Tensor:
    gtf = latent["samples"]
    return gtf


def gtf_to_image(gtf: Tensor) -> Tensor:
    if gtf.shape[1] not in (1, 3, 4):
        raise ValueError("Can only convert a 1, 3 or 4 channel GTF to an image.")
    if gtf.shape[1] == 1:
        gtf = gtf.expand(-1, 3, -1, -1)
    image = gtf.permute(0, 2, 3, 1)
    return image


def gtf_to_mask(gtf: Tensor) -> Tensor:
    if gtf.shape[1] != 1:
        raise ValueError("Cannot convert multi-channel GTF to mask.")
    mask = gtf.squeeze(1)
    return (mask, )


def gtf_update_latent(gtf: Tensor, latent: dict[str, Tensor]) -> dict[str, Tensor]:
    c = gtf.shape[1]
    ec = latent["samples"].shape[1]
    if c != ec:
        raise ValueError(f"Expected GTF with {ec} channels but GTF had {c} channels.")
    updated_latent = copy(latent)
    updated_latent["samples"] = gtf
    return updated_latent
