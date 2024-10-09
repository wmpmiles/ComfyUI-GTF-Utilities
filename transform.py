import torch
import torch.nn.functional as F
from ..gtf_impl.utils import slice_dim, pad_tensor_reflect


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


def component_coloring(tensor: torch.Tensor, diagonals: bool = False) -> torch.Tensor:
    b, c, h, w = tensor.shape
    binary = tensor.to(torch.bool).to(torch.int)
    prepend = torch.zeros(b, c, h, 1, dtype=torch.int)
    diffed = torch.diff(binary, dim=3, prepend=prepend).clamp(0, 1)
    cumsum = torch.cumsum(diffed.flatten(-2), 2).reshape(b, c, h, w)
    coloring_1d = cumsum * binary
    coloring_2d = coloring_1d.clone()
    max_unique = 0
    for bi in range(b):
        for ci in range(c):
            if diagonals:
                equivalences = _equivalences_8_way(bi, ci, binary, coloring_1d)
            else:
                equivalences = _equivalences_4_way(bi, ci, binary, coloring_1d)
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


def _equivalences_4_way(bi: int, ci: int, binary: torch.Tensor, coloring_1d: torch.Tensor) -> torch.Tensor:
        adjacent = torch.logical_and(binary[bi, ci, :-1, :], binary[bi, ci, 1:, :])
        equivalences_u = coloring_1d[bi, ci, :-1, :][adjacent]
        equivalences_d = coloring_1d[bi, ci, 1:, :][adjacent]
        equivalences = torch.stack((equivalences_u, equivalences_d), dim=1)
        return equivalences


def _equivalences_8_way(bi: int, ci: int, binary: torch.Tensor, coloring_1d: torch.Tensor) -> torch.Tensor:
    adjacent_left = torch.logical_and(binary[bi, ci, :-1, :-1], binary[bi, ci, 1:, 1:])
    adjacent_right = torch.logical_and(binary[bi, ci, :-1, 1:], binary[bi, ci, 1:, :-1])
    equ_l_u = coloring_1d[bi, ci, :-1, :-1][adjacent_left]
    equ_r_u = coloring_1d[bi, ci, :-1, 1:][adjacent_right]
    equ_l_d = coloring_1d[bi, ci, 1:, 1:][adjacent_left]
    equ_r_d = coloring_1d[bi, ci, 1:, :-1][adjacent_right]
    equ_m = _equivalences_4_way(bi, ci, binary, coloring_1d)
    equ_l = torch.stack((equ_l_u, equ_l_d), dim=1)
    equ_r = torch.stack((equ_r_u, equ_r_d), dim=1)
    equ = torch.cat((equ_m, equ_l, equ_r), dim=0)
    return equ


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


def gtf_crop_to_bbox(gtf: Tensor, bbox: BoundingBox) -> list[torch.Tensor]:
    wh, lrud = bbox
    w, h = wh
    if gtf.shape[2] != h or gtf.shape[3] != w:
        raise ValueError("GTF dimensions do not match those expected by the bounding box.")
    if gtf.shape[0] != lrud.shape[1] or gtf.shape[1] != lrud.shape[2]:
        raise ValueError("BBOX and GTF channel count and batch size must match")
    cropped = []
    for bi in range(gtf.shape[0]):
        for ci in range(gtf.shape[1]):
            l, r, u, d = lrud[:, bi, ci]
            cropped += [gtf[bi:bi+1, ci:ci+1, u:d, l:r]]
    return cropped


def gtf_uncrop_from_bbox(gtfs: list[Tensor], bbox: BoundingBox) -> Tensor:
    wh, lrud = bbox
    _, b, c = lrud.shape
    if len(gtfs) != b * c:
        raise ValueError("GTFs and BBOX must have same channel count and batch size.")
    batch = []
    for bi in range(lrud.shape[1]):
        channels = []
        for ci in range(lrud.shape[2]):
            gtf = gtfs[bi * c + ci]
            single_lrud = lrud[:, bi, ci]
            uncropped = uncrop_bbox(gtf, single_lrud, wh)
            channels += [uncropped] 
        batch += [torch.cat(channels, dim=1)]
    gtf = torch.cat(batch, dim=0)
    return gtf


def connect_components(gtf: Tensor, diagonals: bool) -> list[Tensor]:
    (coloring, max_unique) = component_coloring(gtf, diagonals)
    if max_unique == 0:
        return [coloring]
    colorings = []
    for i in range(1, max_unique + 1):
        colorings += [(coloring == i).to(gtf.dtype)]
    return colorings
