'''Image bounding-box related functions.

Bounding box is a tuple of two tensors. The first tensor is of shape (2,) and
contains just the width and height of the image in which the bounding box is
placed. The second tensor is of shape (4, batch_size, channel_count) and contains 
the left, right, top, and bottom offsets of the bounding box from the top-left 
of the image for each channel of each image in the batch.
'''


import torch
from torch import Tensor
from math import sqrt
from typing import TypeAlias


BoundingBox: TypeAlias = tuple[Tensor, Tensor]


def gtf_h_min_max(gtf: Tensor) -> Tensor:
    b, c, h, w = gtf.shape
    h_index = torch.arange(0, h).reshape(1, 1, -1).expand(b, c, -1)
    crushed = torch.ceil(torch.clamp(torch.sum(gtf, 3), min=0, max=1))
    bottom_most = torch.argmax(crushed * h_index, 2, keepdim=False) + 1
    top_most = torch.argmax(crushed * torch.flip(h_index, (2, )), 2, keepdim=False)
    return (top_most, bottom_most)


def bbox_from_gtf(gtf: Tensor) -> tuple[Tensor, Tensor]:
    b, c, h, w = gtf.shape
    u, d = gtf_h_min_max(gtf)
    l, r = gtf_h_min_max(gtf.transpose(2, 3))
    hw = Tensor((w, h))
    lrud = torch.stack((l, r, u, d))
    return (hw, lrud)


def lrud_clamp_to_wh(lrud: Tensor, wh: Tensor) -> Tensor:
    w, h = wh
    mins = torch.zeros(lrud.shape)
    maxs = torch.tensor((w, w, h, h))
    clamped = lrud.clamp(min=mins, max=maxs)
    return clamped


def lr_expand(l: Tensor, r: Tensor, w: Tensor, mult: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    cur_len = r - l
    new_len = torch.clamp(torch.round(mult * cur_len), max=w)
    actual_mult = new_len / cur_len
    delta = new_len - cur_len
    new_l = torch.clamp(l - delta // 2, min=0)
    new_r = torch.clamp(new_l + new_len, max=w)
    new_l = new_r - new_len
    return (new_l, new_r, actual_mult)


def lrud_expand(lrud: Tensor, wh: Tensor, area_mult: float) -> Tensor:
    w, h = wh
    side_mult = torch.sqrt(torch.tensor(area_mult))
    l, r, u, d = lrud
    nl, nr, am = lr_expand(l, r, w, side_mult)
    nu, nd, __ = lr_expand(u, d, h, area_mult / am)
    new_lrud = torch.stack((nl, nr, nu, nd))
    return new_lrud


def lrud_outer_square(lrud: Tensor, wh: Tensor) -> Tensor:
    l, r, u, d = lrud
    w, h = wh
    cur_width = r - l
    cur_height = d - u
    w_mul = torch.clamp(cur_height / cur_width, min=1)
    h_mul = torch.clamp(cur_width / cur_height, min=1)
    nl, nr, _ = lr_expand(l, r, w, w_mul) 
    nu, nd, _ = lr_expand(u, d, h, h_mul)
    new_lrud = torch.stack((nl, nr, nu, nd))
    return new_lrud
