import torch
from torch import Tensor


LUMINANCE = torch.tensor((0.2126, 0.7152, 0.0722)).reshape(1, 3, 1, 1)


def to_luminance(gtf: Tensor) -> Tensor:
    if gtf.shape[1] != 3:
        raise ValueErorr("Can only convert a 3 channel GTF to luminance.")
    luminance = (tensor * LUMINANCE).sum(1, keepdims=True)
    return luminance
