import torch
from typing import Literal


F_MAP = {
    "round": torch.round,
    "floor": torch.floor,
    "ceiling": torch.ceil,
}


def to_luminance(tensor: torch.Tensor) -> torch.Tensor:
    LUMINANCE = torch.tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1)
    luminance = (tensor * LUMINANCE).sum(1, keepdims=True)
    return luminance


def quantize_normalized(
    tensor: torch.Tensor,
    steps: int,
    mode: Literal["round", "floor", "ceiling"],
) -> torch.Tensor:
    max_val = steps - 1
    quantized = F_MAP[mode](tensor * max_val) / max_val
    return quantized
