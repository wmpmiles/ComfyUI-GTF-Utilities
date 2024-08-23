import torch


def to_luminance(tensor: torch.Tensor) -> torch.Tensor:
    LUMINANCE = torch.tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1)
    luminance = (tensor * LUMINANCE).sum(1, keepdims=True)
    return luminance


def quantize_normalized(tensor: torch.Tensor, steps: int) -> torch.Tensor:
    max_val = steps - 1
    quantized = torch.round(tensor * max_val) / max_val
    return quantized
