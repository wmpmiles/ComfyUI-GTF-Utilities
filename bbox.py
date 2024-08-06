"""Image bounding-box related functions."""


import torch


def w_min_max_from_mask(mask: torch.Tensor) -> torch.Tensor:
    b, h, w = mask.shape
    h_index = torch.arange(0, w).reshape(1, -1).expand(b, -1)
    crushed = torch.ceil(torch.clamp(torch.sum(mask, 1), min=0, max=1))
    rightmost = torch.argmax(crushed * h_index, 1, keepdim=True) + 1
    leftmost = torch.argmax(crushed * torch.flip(h_index, (1, )), 1, keepdim=True)
    return (leftmost, rightmost)


def bounding_box_from_mask(mask: torch.Tensor) -> torch.Tensor:
    b, h, w = mask.shape
    l, r = w_min_max_from_mask(mask)
    u, d = w_min_max_from_mask(mask.transpose(1, 2))
    w, h = [torch.tensor([[x]]).expand(b, -1) for x in (w, h)]
    bb = torch.cat((l, r, u, d, w, h), 1)
    return bb


def clamp_bounding_box(bbox: torch.Tensor) -> torch.Tensor:
    w = bbox[:,-2].reshape(-1, 1)
    h = bbox[:,-1].reshape(-1, 1)
    mins = torch.zeros(bbox.shape)
    maxs = torch.cat((w, w, h, h, w, h), dim=1)
    clamped = bbox.clamp(min=mins, max=maxs)
    return clamped


def pad_bounding_box(bbox: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    (l, r, u, d) = padding
    term = torch.tensor((l, r, u, d, 0, 0)).reshape(1, -1)
    bb_pad = clamp_bounding_box(bbox + term)
    return bb_pad


def expand_bbox_1d(l, r, w, mult):
    h_len = r - l
    new_h_len = torch.clamp(torch.round(mult * h_len), max=w)
    delta = new_h_len - h_len
    new_l = torch.clamp(l - delta // 2, min=0)
    new_r = torch.clamp(new_l + new_h_len, max=w)
    new_l = new_r - new_h_len
    new_mult = new_h_len / h_len
    return (new_l, new_r, new_mult)


def expand_bounding_box(bbox, area_mult):
    batches = [x.squeeze() for x in bbox.split(1)]
    side_mult = torch.sqrt(torch.tensor(area_mult))
    for i, (l, r, u, d, w, h) in enumerate(batches):
        l, r, m = expand_bbox_1d(l, r, w, side_mult)
        u, d, m = expand_bbox_1d(u, d, h, area_mult / m)
        batches[i] = torch.cat([x.reshape(1, 1) for x in (l, r, u, d, w, h,)], 1)
    bbox = torch.cat(batches, 0)
    return bbox
