"""Image bounding-box related functions."""


import torch
from math import sqrt


def _w_min_max_from_mask(mask: torch.Tensor) -> torch.Tensor:
    b, h, w = mask.shape
    h_index = torch.arange(0, w).reshape(1, -1).expand(b, -1)
    crushed = torch.ceil(torch.clamp(torch.sum(mask, 1), min=0, max=1))
    rightmost = torch.argmax(crushed * h_index, 1, keepdim=True) + 1
    leftmost = \
        torch.argmax(crushed * torch.flip(h_index, (1, )), 1, keepdim=True)
    return (leftmost, rightmost)


def from_mask(
    mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = mask.shape
    l, r = _w_min_max_from_mask(mask)
    u, d = _w_min_max_from_mask(mask.transpose(1, 2))
    hw = torch.tensor((w, h))
    lrud = torch.cat((l, r, u, d), 1)
    bb = (hw, lrud)
    return bb


def clamp_lrud(lrud: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
    w = wh[0].reshape(1, 1)
    h = wh[1].reshape(1, 1)
    mins = torch.zeros(lrud.shape)
    maxs = torch.cat((w, w, h, h), dim=1)
    clamped = lrud.clamp(min=mins, max=maxs)
    return clamped


def pad(
    bbox: torch.Tensor,
    padding: tuple[int, int, int, int]
) -> torch.Tensor:
    wh, lrud = bbox
    (l, r, u, d) = padding
    term = torch.tensor((l, r, u, d)).reshape(1, -1)
    padded = clamp_lrud(lrud + term, wh)
    return (wh, padded)


def _expand_1d(
    l: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    mult: float
) -> tuple[int, int, float]:
    h_len = r - l
    new_h_len = torch.clamp(torch.round(mult * h_len), max=w)
    delta = new_h_len - h_len
    new_l = torch.clamp(l - delta // 2, min=0)
    new_r = torch.clamp(new_l + new_h_len, max=w)
    new_l = new_r - new_h_len
    new_mult = new_h_len / h_len
    return (new_l, new_r, new_mult)


def expand_lrud(
    lrud: torch.Tensor,
    wh: torch.Tensor,
    area_mult: float
) -> torch.Tensor:
    w, h = wh
    unbatched = (x.squeeze() for x in lrud.split(1))
    expanded = []
    side_mult = torch.sqrt(torch.tensor(area_mult))
    for l, r, u, d in unbatched:
        l, r, m = _expand_1d(l, r, w, side_mult)
        u, d, m = _expand_1d(u, d, h, area_mult / m)
        expanded += [torch.tensor((l, r, u, d)).reshape(1, -1)]
    new_lrud = torch.cat(expanded, 0)
    return new_lrud


def expand_lrud_square(
    lrud: torch.Tensor,
    wh: torch.Tensor,
    area_mult: float
) -> torch.Tensor:
    w, h = wh
    unbatched = [x.squeeze() for x in lrud.split(1)]
    expanded = []
    for l, r, u, d in unbatched:
        width = r - l
        height = d - u
        to_square = max(width, height) / min(width, height)
        small_mult = min(to_square, area_mult)
        if width > height:
            u, d, _ = _expand_1d(u, d, h, small_mult)
        else:
            l, r, _ = _expand_1d(l, r, w, small_mult)
        rem_mult = area_mult / small_mult
        if rem_mult > 1:
            side_mult = sqrt(rem_mult)
            l, r, m = _expand_1d(l, r, w, side_mult)
            u, d, _ = _expand_1d(u, d, h, rem_mult / m)
        expanded += [torch.tensor((l, r, u, d)).reshape(1, -1)]
    new_lrud = torch.cat(expanded, 0)
    return new_lrud
