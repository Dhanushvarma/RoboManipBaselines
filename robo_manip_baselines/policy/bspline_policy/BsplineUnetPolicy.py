"""Diffusion U-Net backbone predicting B-spline parameters instead of actions.

BSP is an action *representation*, not a network, so this is a thin subclass of
RMB's existing Diffusion Policy backbone. Two changes, mirroring
``DiffusionUnetBSplineImagePolicy`` in the reference implementation:

1. **One extra action channel** for the knot column. The caller passes the
   physical action dimension (7 for a UR5e with gripper); the model internally
   works in ``1 + action_dim`` (8).
2. **Return the whole horizon.** The stock policy slices
   ``action_pred[:, To-1 : To-1+n_action_steps]`` to hand back a chunk of
   executable actions. A B-spline segment is not a chunk -- every one of its
   ``chunk_size + 2*degree`` rows is needed to evaluate the curve -- so the full
   matrix is returned.

Note that the reference implementation overrides a ``select_action`` hook that
exists in the B-spline authors' fork of Diffusion Policy but *not* in the fork
RMB vendors (``third_party/diffusion_policy``). Overriding ``predict_action``
achieves the same thing against RMB's fork, and is what the reference's
transformer variant does anyway.
"""

import copy
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)

from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (  # noqa: E402
    DiffusionUnetHybridImagePolicy,
)


def _widen_action_shape(shape_meta):
    """Add the knot channel to a copy of shape_meta."""
    bspline_shape_meta = copy.deepcopy(shape_meta)
    action_shape = bspline_shape_meta["action"]["shape"]
    if len(action_shape) != 1:
        raise ValueError(
            f"[BsplineUnetPolicy] Expected a 1-D action shape, got {action_shape}"
        )
    bspline_shape_meta["action"]["shape"] = [int(action_shape[0]) + 1]
    return bspline_shape_meta, int(action_shape[0])


class BsplineUnetPolicy(DiffusionUnetHybridImagePolicy):
    """Diffusion U-Net over B-spline parameters."""

    def __init__(self, shape_meta: dict, bspline_degree: int = 3, **kwargs):
        bspline_shape_meta, regular_action_dim = _widen_action_shape(shape_meta)
        super().__init__(shape_meta=bspline_shape_meta, **kwargs)
        self.regular_action_dim = regular_action_dim
        self.bspline_degree = int(bspline_degree)

    def predict_action(self, obs_dict):
        result = super().predict_action(obs_dict)
        # The full (B, chunk_size + 2*degree, 1 + action_dim) parameter matrix;
        # the base class's chunk slice is meaningless for a spline segment.
        result["action"] = result["action_pred"]
        return result
