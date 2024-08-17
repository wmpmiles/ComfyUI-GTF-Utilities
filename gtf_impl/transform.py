import torch
import torch.nn.functional as F
from gtf_impl.utils import slice_dim, pad_tensor_reflect


def crop_uncrop(
    tensor: torch.Tensor,
    dim: int,
    new_length: int,
    anchor: str,
    mode: str,
) -> torch.Tensor:
    length = int(tensor.shape[dim])
    delta = new_length - length
    match anchor:
        case "left": deltas = (0, delta)
        case "right": deltas = (delta, 0)
        case "middle": deltas = (delta // 2, delta - delta // 2)
        case _: raise ValueError("anchor must be one of [left, right, middle]")
    pad_cropped = _pad_crop_dim(tensor, dim, deltas, mode)
    return pad_cropped


def _pad_crop_dim(
    tensor: torch.Tensor,
    dim: int,
    deltas: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    dl, dr = deltas
    sl, sr = (min(0, dl), min(0, dr))
    pl, pr = (max(0, dl), max(0, dr))
    cropped = tensor.clone()[slice_dim(dim, -sl, sr if sr else None)]
    padded = _pad_dim(cropped, dim, (pl, pr), mode)
    return padded


def _pad_dim(
    tensor: torch.Tensor,
    dim: int,
    pad: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    match mode:
        case "zero":
            dims = tensor.dim()
            pad_list = [0, 0] * (dims - dim - 1) + list(pad)
            padded = F.pad(tensor, pad_list)
        case "reflect":
            padded = pad_tensor_reflect(tensor, dim, pad[0], pad[1])
        case _:
            raise ValueError(f"Unknown mode: {mode}")
    return padded


def uncrop_bbox(
    tensor: torch.Tensor,
    single_lrud: torch.Tensor,
    wh: torch.Tensor
) -> torch.Tensor:
    w, h = wh
    b, c, th, tw = (int(x) for x in tensor.shape)
    l, r, u, d = (int(x) for x in single_lrud)
    bh, bw = (d - u, r - l)
    if th != bh or tw != bw:
        raise ValueError("GTF dimensions don't match bounding box dimensions.")
    uncropped = torch.zeros(b, c, h, w)
    uncropped[:, :, u:d, l:r] = tensor
    return uncropped


def component_coloring(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.shape
    binary = tensor.clamp(0, 1).round().to(torch.int)
    prepend = torch.zeros(b, c, h, 1, dtype=torch.int)
    diffed = torch.diff(binary, dim=3, prepend=prepend).clamp(0, 1)
    cumsum = torch.cumsum(diffed.flatten(-2), 2).reshape(b, c, h, w)
    coloring_1d = cumsum * binary
    coloring_2d = coloring_1d.clone()
    max_unique = 0
    for bi in range(b):
        for ci in range(c):
            adjacent = torch.logical_and(
                binary[bi, ci, :-1, :],
                binary[bi, ci, 1:, :]
            )
            equivalences_u = coloring_1d[bi, ci, :-1, :][adjacent]
            equivalences_l = coloring_1d[bi, ci, 1:, :][adjacent]
            equivalences = torch.stack((equivalences_u, equivalences_l), dim=1)
            unique = torch.unique_consecutive(cumsum[bi, ci]).tolist()
            if unique[0] == 0:
                unique = unique[1:]
            if len(unique) == 0:
                coloring_2d[bi, ci] = 0
                continue
            umax = int(unique[-1])
            parent = list(range(umax + 1))
            size = [1] * (umax + 1)
            for pair in equivalences.tolist():
                p0, p1 = pair
                _disjoint_set_union(p0, p1, parent, size)
            for x in range(umax + 1):
                parent[x] = _disjoint_set_find(x, parent)
            # relabel
            parent = torch.tensor(parent)
            parent_unique = torch.unique(parent)
            max_unique = max(max_unique, int(parent_unique.shape[0]) - 1)
            index_map = torch.zeros((umax + 1, ), dtype=torch.long)
            index_map[parent_unique] = torch.arange(0, parent_unique.shape[0])
            parent_relabel = index_map.index_select(0, parent)
            # replace
            c1d = coloring_2d[bi, ci]
            c1d.reshape(-1)[:] = \
                parent_relabel.index_select(0, c1d.reshape(-1))
    return (coloring_2d, max_unique)


def _disjoint_set_union(x, y, parents, size):
    px = _disjoint_set_find(x, parents)
    py = _disjoint_set_find(y, parents)
    if px != py:
        if size[px] < size[py]:
            (px, py) = (py, px)
        parents[py] = px
        size[px] += size[py]


def _disjoint_set_find(x, parents):
    while parents[x] != x:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x
