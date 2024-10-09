"""
Resampling related functions.

Supports:
- Nearest-neighbor resampling (1D, seperable)
- Bilinear resampling (1D, seperable)
- Area resampling (1D, seperable)
- Lanczos resampling (1D, 2D, seperable)
- Mitchell-Netravali resampling (1D, 2D, seperable)
"""

import torch
from torch import Tensor
import math
from typing import Callable
import utils as U


#                              #
# NEAREST NEIGHBOUR RESAMPLING #
#                              #

def _nearest_neighbor_indices(
    I_t: Tensor,
    l: int,
    L: int
) -> Tensor:
    i_snn = ((2 * I_t + 1) * l) // (2 * L)
    return i_snn


def _nearest_neighbor_1d(
    d_s: Tensor,
    L: int,
    dim: int = 0
) -> Tensor:
    L = L
    l = int(d_s.shape[dim])
    I_t = torch.arange(L)
    indices = _nearest_neighbor_indices(I_t, l, L)
    downsampled = d_s.index_select(dim, indices)
    return downsampled


def nearest_neighbor_2d(
    d_s: Tensor,
    L: tuple[int, int],
    dim: tuple[int, int] = (0, 1)
) -> Tensor:
    D_0 = _nearest_neighbor_1d(d_s, L[0], dim[0])
    D_1 = _nearest_neighbor_1d(D_0, L[1], dim[1])
    return D_1


#                   #
# FILTER RESAMPLING #
#                   #

def _window_lengths(
    l: int,
    L: int,
    radius: int,
) -> int:
    if radius == 0:
        w, W = 1, 1
    elif L <= l:
        W = 2 * radius
        w = -((-l * W) // L)
    else:
        w = 2 * radius
        W = -((-L * w) // l)
    return (w, W)


def _padding_1d(
    i_snn: Tensor,
    I_t: Tensor,
    l: int,
    L: int,
    W: int,
    w: int,
) -> Tensor:
    if w == 1 and W == 1:
        p = i_snn * 0
    elif L <= l:
        p = (W * l + L * (2 * i_snn + 1) - l * (2 * I_t + 1)) // (2 * L)
    else:
        c = l * (2 * I_t + 1) > L * (2 * i_snn + 1)
        p = (w // 2) - c.to(torch.int)
    return p


def _x_values_1d(
    i_snn: Tensor,
    i_w: Tensor,
    I_t: Tensor,
    l: int,
    L: int,
    p: Tensor
) -> Tensor:
    X_w_n = U.outer_sum(
        (2 * (i_snn - p) + 1) * L - (2 * I_t + 1) * l, (2 * L) * i_w
    )
    if L <= l:
        x_values = X_w_n / (2 * l)
    else:
        # breaks notation convention
        x_values = X_w_n / (2 * L)
    return x_values


def _data_values_1d(
    d_p: Tensor,
    i_snn: Tensor,
    i_w: Tensor,
    p: Tensor,
    p0: int,
    L: int,
    w: int,
    dim: int = 0
) -> Tensor:
    o = i_snn - p + p0
    i_2d = U.outer_sum(o, i_w)
    i_1d = i_2d.flatten()
    d_1d = d_p.index_select(dim, i_1d)
    shape = list(d_p.shape)
    shape = shape[:dim] + [L, w] + shape[dim+1:]
    d_2d = d_1d.reshape(*shape)
    return d_2d


def filter_1d(
    d_s: Tensor,
    L: int,
    radius: int,
    filter: Callable[[Tensor], Tensor],
    dim: int = 0,
) -> Tensor:
    l = int(d_s.shape[dim])
    (w, W) = _window_lengths(l, L, radius)
    I_t = torch.arange(L)
    i_snn = _nearest_neighbor_indices(I_t, l, L)
    p = _padding_1d(i_snn, I_t, l, L, W, w)
    i_w = torch.arange(w)
    x_values = _x_values_1d(i_snn, i_w, I_t, l, L, p)
    p_0 = int(p[0])
    d_p = U.pad_tensor_reflect(d_s, dim, p_0, p_0 + 1)
    d_2d = _data_values_1d(d_p, i_snn, i_w, p, p_0, L, w, dim)
    f = filter(x_values)
    f_n = U.sum_normalize(f, (1,))
    shape = [1] * d_2d.dim()
    shape[dim] = L
    shape[dim+1] = w
    d_f = d_2d * f_n.reshape(*shape)
    d_r = d_f.sum(dim+1)
    return d_r


def filter_2d_seperable(
    d_s: Tensor,
    L: tuple[int, int],
    radius: int,
    filter: Callable[[Tensor], Tensor],
    dim: tuple[int, int] = (0, 1),
) -> Tensor:
    D_0 = filter_1d(d_s, L[0], radius, filter, dim[0])
    D_1 = filter_1d(D_0, L[1], radius, filter, dim[1])
    return D_1


def _x_values_2d(
    X_w: tuple[Tensor, Tensor],
    L: tuple[int, int],
    w: tuple[int, int],
) -> Tensor:
    X_w_sqr = (X_w[0] * X_w[0], X_w[1] * X_w[1])
    X_w2_sqr = X_w_sqr[0].reshape(L[0], 1, w[0], 1) + \
        X_w_sqr[1].reshape(1, L[1], 1, w[1])
    X_w2 = torch.sqrt(X_w2_sqr)
    return X_w2


def _data_values_2d(
    d_p: Tensor,
    i_snn: tuple[Tensor, Tensor],
    p: tuple[Tensor, Tensor],
    i_w: tuple[Tensor, Tensor],
    p0: tuple[int, int],
    w: tuple[int, int],
    L: tuple[int, int],
    l_p: tuple[int, int],
    dim: tuple[int, int],
) -> Tensor:
    o = (i_snn[0] - p[0] + p0[0], i_snn[1] - p[1] + p0[1])
    i_2d = (U.outer_sum(o[0], i_w[0]), U.outer_sum(o[1], i_w[1]))
    i_4d = (i_2d[0] * l_p[1]).reshape(L[0], 1, w[0], 1) + \
        i_2d[1].reshape(1, L[1], 1, w[1])
    i_1d = i_4d.flatten()
    d_p = d_p.transpose(dim[1], -1)
    d_p = d_p.transpose(dim[0], -2)
    d_1d = d_p.flatten(start_dim=-2, end_dim=-1).index_select(-1, i_1d)
    shape = list(d_1d.shape)[:-1]
    d_4d = d_1d.reshape(*shape, L[0], L[1], w[0], w[1])
    d_4d = d_4d.transpose(dim[0], -4)
    d_4d = d_4d.transpose(dim[1], -3)
    return d_4d


def filter_2d(
    d_s: Tensor,
    L: tuple[int, int],
    radius: int,
    filter: Callable[[Tensor], Tensor],
    dim: tuple[int, int] = (0, 1),
) -> Tensor:
    l = (int(d_s.shape[dim[0]]), int(d_s.shape[dim[1]]))
    (w, W) = tuple(zip(
        _window_lengths(l[0], L[0], radius),
        _window_lengths(l[1], L[1], radius)
    ))
    I_t = (torch.arange(L[0]), torch.arange(L[1]))
    i_snn = (
        _nearest_neighbor_indices(I_t[0], l[0], L[0]),
        _nearest_neighbor_indices(I_t[1], l[1], L[1])
    )
    p = (
        _padding_1d(i_snn[0], I_t[0], l[0], L[0], W[0], w[0]),
        _padding_1d(i_snn[1], I_t[1], l[1], L[1], W[1], w[0])
    )
    i_w = (torch.arange(w[0]), torch.arange(w[1]))
    X_w = (
        _x_values_1d(i_snn[0], i_w[0], I_t[0], l[0], L[0], p[0]),
        _x_values_1d(i_snn[1], i_w[1], I_t[1], l[1], L[1], p[1])
    )
    X_w2 = _x_values_2d(X_w, L, w)
    p0 = (int(p[0][0]), int(p[1][0]))
    d_p = U.pad_tensor_reflect(d_s, dim[0], p0[0], p0[0] + 1)
    d_p = U.pad_tensor_reflect(d_p, dim[1], p0[1], p0[1] + 1)
    l_p = (int(d_p.shape[dim[0]]), int(d_p.shape[dim[1]]))
    d_4d = _data_values_2d(d_p, i_snn, p, i_w, p0, w, L, l_p, dim)
    f = filter(X_w2)
    f_n = U.sum_normalize(f, (2, 3))
    shape = [1] * d_4d.dim()
    shape[dim[0]] = L[0]
    shape[dim[1]] = L[1]
    shape[-2] = w[0]
    shape[-1] = w[1]
    d_f = d_4d * f_n.reshape(*shape)
    d_r = d_f.sum(-1).sum(-1)
    return d_r
