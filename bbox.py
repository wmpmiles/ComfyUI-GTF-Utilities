"""Image bounding-box related functions."""


import torch
import torch.nn.functional as F
from math import sqrt


def w_min_max_from_mask(mask: torch.Tensor) -> torch.Tensor:
    b, h, w = mask.shape
    h_index = torch.arange(0, w).reshape(1, -1).expand(b, -1)
    crushed = torch.ceil(torch.clamp(torch.sum(mask, 1), min=0, max=1))
    rightmost = torch.argmax(crushed * h_index, 1, keepdim=True) + 1
    leftmost = torch.argmax(crushed * torch.flip(h_index, (1, )), 1, keepdim=True)
    return (leftmost, rightmost)


def bounding_box_from_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = mask.shape
    l, r = w_min_max_from_mask(mask)
    u, d = w_min_max_from_mask(mask.transpose(1, 2))
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


def pad_bounding_box(bbox: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    wh, lrud = bbox
    (l, r, u, d) = padding
    term = torch.tensor((l, r, u, d)).reshape(1, -1)
    padded = clamp_lrud(lrud + term, wh)
    return (wh, padded)


def expand_bbox_1d(l, r, w, mult):
    h_len = r - l
    new_h_len = torch.clamp(torch.round(mult * h_len), max=w)
    delta = new_h_len - h_len
    new_l = torch.clamp(l - delta // 2, min=0)
    new_r = torch.clamp(new_l + new_h_len, max=w)
    new_l = new_r - new_h_len
    new_mult = new_h_len / h_len
    return (new_l, new_r, new_mult)


def expand_lrud(lrud: torch.Tensor, wh: torch.Tensor, area_mult: float) -> torch.Tensor:
    w, h = wh
    unbatched = (x.squeeze() for x in lrud.split(1))
    expanded = []
    side_mult = torch.sqrt(torch.tensor(area_mult))
    for l, r, u, d in unbatched:
        l, r, m = expand_bbox_1d(l, r, w, side_mult)
        u, d, m = expand_bbox_1d(u, d, h, area_mult / m)
        expanded += [torch.tensor((l, r, u, d)).reshape(1, -1)]
    new_lrud = torch.cat(expanded, 0)
    return new_lrud


def expand_lrud_square(lrud: torch.Tensor, wh: torch.Tensor, area_mult: float) -> torch.Tensor:
    w, h = wh
    unbatched = [x.squeeze() for x in lrud.split(1)]
    expanded = []
    for l, r, u, d in unbatched:
        width = r - l
        height = d - u
        to_square = max(width, height) / min(width, height)
        small_mult = min(to_square, area_mult)
        if width > height:
            u, d, _ = expand_bbox_1d(u, d, h, small_mult)
        else:
            l, r, _ = expand_bbox_1d(l, r, w, small_mult)
        rem_mult = area_mult / small_mult
        if rem_mult > 1:
            side_mult = sqrt(rem_mult)
            l, r, m = expand_bbox_1d(l, r, w, side_mult)
            u, d, _ = expand_bbox_1d(u, d, h, rem_mult / m)
        expanded += [torch.tensor((l, r, u, d)).reshape(1, -1)]
    new_lrud = torch.cat(expanded, 0)
    return new_lrud


def component_coloring(tensor: torch.Tensor) -> torch.Tensor:
    from time import perf_counter
    s1 = perf_counter()
    binary = tensor.clamp(0, 1).round()
    b, c, h, w = binary.shape
    prepend = torch.zeros(b, c, h, 1)
    diffed = torch.diff(binary, dim=3, prepend=prepend).clamp(0, 1)
    cumsum = torch.cumsum(diffed, 3)
    row_indices = torch.arange(0, h) 
    offset_base = ((w + 1) // 2) + 1
    row_offsets = row_indices * offset_base
    cumsum_offset = (cumsum + row_offsets.reshape(1, 1, -1, 1))
    coloring_1d =  cumsum_offset * binary
    binary_unfolded = F.unfold(binary, (2, 1))
    coloring_unfolded = F.unfold(coloring_1d, (2, 1))
    adjacent = torch.logical_and(binary_unfolded[:,0], binary_unfolded[:,1])
    adjacent_mask = adjacent.reshape(1, 1, -1).expand(b, 2 * c, -1)
    equivalences = coloring_unfolded[adjacent_mask].reshape(b, c, 2, -1)
    coloring_2d = coloring_1d.clone()
    for b, cumsum_b, equiv_b in zip(coloring_2d.split(1), cumsum_offset.split(1), equivalences.split(1)):
        for c, cumsum_c, equiv_c in zip(b.squeeze(0).split(1), cumsum_b.squeeze(0).split(1), equiv_b.squeeze(0).split(1)):
            unique_c = torch.unique_consecutive(cumsum_c.squeeze(0))
            unique = unique_c[unique_c % offset_base != 0].to(torch.int)
            #unique = [int(x) for x in unique_filtered]
            umax = int(unique[-1])
            parent = list(range(umax + 1))
            # TODO: test unique equiv
            for pair in ([int(y) for y in x.squeeze(0)] for x in equiv_c.squeeze(0).T.split(1)):
                if parent[pair[1]] != pair[1]:
                    parent[pair[0]] = parent[pair[1]]
                else:
                    parent[pair[1]] = parent[pair[0]]
            # relabel, TODO: look at index_select based (unique on parent)
            label = 1
            for u in unique:
                u = int(u)
                if parent[u] == u:
                    parent[u] = label
                    label += 1
                else:
                    parent[u] = parent[parent[u]]
            # replace
            index_map = torch.zeros((umax + 1, ))
            index_map[unique] = torch.tensor(parent)[unique].to(torch.float)
            c.reshape(-1)[:] = index_map.index_select(0, c.reshape(-1).to(torch.int32))
    s2 = perf_counter()
    print(s2 - s1)
    return coloring_2d


a = torch.randn(1, 1, 1000, 1000).clamp(0, 1).round() 
b = torch.tensor([[[[1, 0, 1,],[1, 1, 1]]]]).to(torch.float)
print(component_coloring(a[0:1, 0:1, :1000, :1000])[0, 0, :5, :10])
print(component_coloring(a[0:1, 0:1, :1000, :1000])[0, 0, -5:, -5:])
print(component_coloring(b))