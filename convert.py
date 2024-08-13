import torch


def to_luminance(tensor: torch.Tensor) -> torch.Tensor:
    LUMINANCE = torch.tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1)
    luminance = (tensor * LUMINANCE).sum(2, True)
    return luminance
