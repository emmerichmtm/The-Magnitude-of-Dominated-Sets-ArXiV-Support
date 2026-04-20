
"""
exact_magnitude_hypervolume.py

Exact 2D/3D hypervolume and magnitude for maximization with a given anchor point,
together with tie-shared inclusion-exclusion subgradients.

The implementation is intended for small and medium-sized point sets when one
wants exact values and exact piecewise-linear / piecewise-polynomial gradient
information for research experiments.

Definitions
-----------
For a finite approximation set P = {p^(1), ..., p^(n)} in R^d with anchor point a
and maximization, define the translated points

    q^(i) = p^(i) - a,

assuming q^(i) >= 0 componentwise.

The dominated region is the union of anchored boxes
    D(P) = union_i [0, q^(i)].

Hypervolume:
    HV_d(P; a) = vol_d(D(P))

computed exactly by inclusion-exclusion.

Magnitude:
For d = 2:
    Mag_2(P; a) = 1 + 1/2 (L_x + L_y) + 1/4 HV_2(P; a)

For d = 3:
    Mag_3(P; a) = 1
                  + 1/2 (L_x + L_y + L_z)
                  + 1/4 (A_xy + A_xz + A_yz)
                  + 1/8 HV_3(P; a)

where L_* are axis projection lengths, and A_xy etc. are 2D dominated areas of
the coordinate projections. These are the dominated-set formulas used in the
report in the l1 box setting.

Subgradients:
The exact inclusion-exclusion gradient is not unique at ties. We return a
tie-shared subgradient: when multiple points attain a coordinatewise minimum in a
subset term, that term's derivative is split equally among the tied points.
Likewise, maxima in the axis-projection terms are shared equally among ties.

Author: OpenAI assistant, prepared for Michael Emmerich.
License: MIT-style; feel free to adapt.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[Sequence[float]]


def _as_array(points: ArrayLike, dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError("points must be a 2D array-like object of shape (n_points, dimension)")
    if dim is not None and arr.shape[1] != dim:
        raise ValueError(f"expected points in dimension {dim}, got {arr.shape[1]}")
    return arr


def _translate_and_validate(points: ArrayLike, anchor: Sequence[float], dim: Optional[int] = None) -> np.ndarray:
    pts = _as_array(points, dim=dim)
    a = np.asarray(anchor, dtype=float)
    if a.ndim != 1 or a.shape[0] != pts.shape[1]:
        raise ValueError("anchor must be a 1D array-like object with the same dimension as the points")
    q = pts - a
    if np.any(q < -1e-12):
        raise ValueError("all points must weakly dominate the anchor point componentwise")
    q[q < 0.0] = 0.0
    return q


def _prod_except(mins: np.ndarray, k: int) -> float:
    if mins.size == 1:
        return 1.0
    prod = 1.0
    for idx, val in enumerate(mins):
        if idx != k:
            prod *= float(val)
    return prod


def exact_hypervolume_max(points: ArrayLike, anchor: Sequence[float]) -> float:
    """
    Exact hypervolume for maximization with a given anchor point, in any dimension.
    Complexity is O(2^n * d) via inclusion-exclusion, intended for exact studies.
    """
    q = _translate_and_validate(points, anchor)
    n, d = q.shape
    total = 0.0
    for r in range(1, n + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for subset in combinations(range(n), r):
            mins = np.min(q[list(subset), :], axis=0)
            total += sign * float(np.prod(mins))
    return total


def exact_hypervolume_gradient_max(points: ArrayLike, anchor: Sequence[float]) -> np.ndarray:
    """
    Tie-shared exact inclusion-exclusion subgradient of hypervolume for maximization.

    Returns an array of shape (n_points, dimension).
    """
    q = _translate_and_validate(points, anchor)
    n, d = q.shape
    grad = np.zeros((n, d), dtype=float)

    for r in range(1, n + 1):
        sign = 1.0 if (r % 2 == 1) else -1.0
        for subset in combinations(range(n), r):
            sub = q[list(subset), :]
            mins = np.min(sub, axis=0)
            for k in range(d):
                min_val = mins[k]
                tied_positions = [pos for pos, idx in enumerate(subset) if abs(q[idx, k] - min_val) <= 1e-12]
                if not tied_positions:
                    continue
                coeff = sign * _prod_except(mins, k) / float(len(tied_positions))
                for pos in tied_positions:
                    idx = subset[pos]
                    grad[idx, k] += coeff
    return grad


def hypervolume_2d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0)) -> float:
    return exact_hypervolume_max(points, anchor)


def hypervolume_3d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0, 0.0)) -> float:
    return exact_hypervolume_max(points, anchor)


def hypervolume_gradient_2d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    return exact_hypervolume_gradient_max(points, anchor)


def hypervolume_gradient_3d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    return exact_hypervolume_gradient_max(points, anchor)


def _axis_max_gradient(q: np.ndarray) -> np.ndarray:
    """
    Gradient of sum_k max_i q_{ik}, with ties shared equally.
    """
    n, d = q.shape
    grad = np.zeros((n, d), dtype=float)
    for k in range(d):
        m = float(np.max(q[:, k]))
        tied = np.where(np.abs(q[:, k] - m) <= 1e-12)[0]
        if tied.size:
            grad[tied, k] += 1.0 / float(tied.size)
    return grad


def magnitude_2d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0)) -> float:
    q = _translate_and_validate(points, anchor, dim=2)
    lx = float(np.max(q[:, 0])) if len(q) else 0.0
    ly = float(np.max(q[:, 1])) if len(q) else 0.0
    hv = exact_hypervolume_max(q, (0.0, 0.0))
    return 1.0 + 0.5 * (lx + ly) + 0.25 * hv


def magnitude_gradient_2d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    q = _translate_and_validate(points, anchor, dim=2)
    axis_grad = _axis_max_gradient(q)
    hv_grad = exact_hypervolume_gradient_max(q, (0.0, 0.0))
    return 0.5 * axis_grad + 0.25 * hv_grad


def magnitude_3d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0, 0.0)) -> float:
    q = _translate_and_validate(points, anchor, dim=3)
    lx = float(np.max(q[:, 0])) if len(q) else 0.0
    ly = float(np.max(q[:, 1])) if len(q) else 0.0
    lz = float(np.max(q[:, 2])) if len(q) else 0.0

    area_xy = exact_hypervolume_max(q[:, [0, 1]], (0.0, 0.0))
    area_xz = exact_hypervolume_max(q[:, [0, 2]], (0.0, 0.0))
    area_yz = exact_hypervolume_max(q[:, [1, 2]], (0.0, 0.0))
    hv3 = exact_hypervolume_max(q, (0.0, 0.0, 0.0))

    return 1.0 + 0.5 * (lx + ly + lz) + 0.25 * (area_xy + area_xz + area_yz) + 0.125 * hv3


def magnitude_gradient_3d_max(points: ArrayLike, anchor: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    q = _translate_and_validate(points, anchor, dim=3)
    n = q.shape[0]

    axis_grad = _axis_max_gradient(q)

    hv3_grad = exact_hypervolume_gradient_max(q, (0.0, 0.0, 0.0))

    hv_xy = exact_hypervolume_gradient_max(q[:, [0, 1]], (0.0, 0.0))
    hv_xz = exact_hypervolume_gradient_max(q[:, [0, 2]], (0.0, 0.0))
    hv_yz = exact_hypervolume_gradient_max(q[:, [1, 2]], (0.0, 0.0))

    proj_grad = np.zeros((n, 3), dtype=float)
    proj_grad[:, 0] += hv_xy[:, 0]
    proj_grad[:, 1] += hv_xy[:, 1]
    proj_grad[:, 0] += hv_xz[:, 0]
    proj_grad[:, 2] += hv_xz[:, 1]
    proj_grad[:, 1] += hv_yz[:, 0]
    proj_grad[:, 2] += hv_yz[:, 1]

    return 0.5 * axis_grad + 0.25 * proj_grad + 0.125 * hv3_grad


def normalize_rows(grad: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize each pointwise gradient vector to unit Euclidean norm when nonzero.
    Useful for the projected normalized-gradient methods described in the paper.
    """
    g = np.asarray(grad, dtype=float).copy()
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    mask = norms > eps
    g[mask[:, 0]] /= norms[mask]
    return g


def simplex_project_rows(points: np.ndarray, total: float = 1.0) -> np.ndarray:
    """
    Project each row onto the standard simplex {x >= 0, sum x = total}.
    """
    X = np.asarray(points, dtype=float)
    Y = np.zeros_like(X)
    for i, v in enumerate(X):
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - total
        rho = np.nonzero(u - cssv / (np.arange(len(u)) + 1) > 0)[0]
        if len(rho) == 0:
            theta = 0.0
        else:
            rho = rho[-1]
            theta = cssv[rho] / (rho + 1.0)
        Y[i] = np.maximum(v - theta, 0.0)
    return Y


def projected_gradient_step(
    points: ArrayLike,
    gradient_fn: Callable[[ArrayLike, Sequence[float]], np.ndarray],
    anchor: Sequence[float],
    step_size: float,
    projector: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    normalize_pointwise: bool = True,
) -> np.ndarray:
    """
    One projected ascent step:
        P_new = Pi( P + alpha * G(P) )

    where G may be normalized row-wise if desired.
    """
    P = _as_array(points)
    G = np.asarray(gradient_fn(P, anchor), dtype=float)
    if normalize_pointwise:
        G = normalize_rows(G)
    P_new = P + step_size * G
    if projector is not None:
        P_new = projector(P_new)
    return P_new


def _demo() -> None:
    print("=== 2D example ===")
    P2 = np.array([(1.0, 3.0), (3.0, 2.0), (5.0, 1.0)])
    a2 = (0.0, 0.0)
    print("HV_2 =", hypervolume_2d_max(P2, a2))
    print("Mag_2 =", magnitude_2d_max(P2, a2))
    print("grad HV_2 =\n", hypervolume_gradient_2d_max(P2, a2))
    print("grad Mag_2 =\n", magnitude_gradient_2d_max(P2, a2))

    print("\n=== 3D example ===")
    P3 = np.array(
        [
            (1.0, 6.0, 4.0),
            (3.0, 5.0, 1.0),
            (4.0, 4.0, 6.0),
            (5.0, 2.0, 3.0),
            (6.0, 1.0, 5.0),
            (1.0, 3.0, 7.0),
            (2.0, 2.0, 8.0),
        ]
    )
    a3 = (0.0, 0.0, 0.0)
    print("HV_3 =", hypervolume_3d_max(P3, a3))
    print("Mag_3 =", magnitude_3d_max(P3, a3))
    print("grad HV_3 =\n", hypervolume_gradient_3d_max(P3, a3))
    print("grad Mag_3 =\n", magnitude_gradient_3d_max(P3, a3))


if __name__ == "__main__":
    _demo()
