"""Lightweight B-spline knot encoding helpers.

Vendored verbatim from the B-spline Policy reference implementation
(``bspline_policy/common/knots.py``). Only used when ``relative_knots=True``;
every shipped BSP config sets it to ``False``.

Both functions are torch/numpy polymorphic and operate on the last axis
(``[..., 0]`` selects the knot column of a ``(..., L, 1 + action_dim)`` matrix).
They mutate through a view, which is why the originals are copied first.
"""

import torch


def encode_relative_knots(action_data, degree: int = 3):
    """Encode knot values as first valid knot plus adjacent differences."""
    result = action_data.clone() if torch.is_tensor(action_data) else action_data.copy()
    knots = result[..., 0]
    original_knots = knots.clone() if torch.is_tensor(knots) else knots.copy()

    knots[..., 0] = original_knots[..., degree]
    knots[..., 1:] = original_knots[..., 1:] - original_knots[..., :-1]
    return result


def decode_relative_knots(action_data, degree: int = 3):
    """Decode the representation produced by encode_relative_knots."""
    result = action_data.clone() if torch.is_tensor(action_data) else action_data.copy()
    encoded = (
        result[..., 0].clone() if torch.is_tensor(result) else result[..., 0].copy()
    )
    knots = result[..., 0]
    n_knots = knots.shape[-1]

    knots[..., degree] = encoded[..., 0]
    for knot_idx in range(degree - 1, -1, -1):
        knots[..., knot_idx] = knots[..., knot_idx + 1] - encoded[..., knot_idx + 1]
    for knot_idx in range(degree + 1, n_knots):
        knots[..., knot_idx] = knots[..., knot_idx - 1] + encoded[..., knot_idx]

    return result
