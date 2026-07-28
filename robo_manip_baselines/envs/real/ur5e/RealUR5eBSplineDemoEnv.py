import numpy as np

from .RealUR5eBSplineEnvBase import RealUR5eBSplineEnvBase


class RealUR5eBSplineDemoEnv(RealUR5eBSplineEnvBase):
    def __init__(self, **kwargs):
        RealUR5eBSplineEnvBase.__init__(
            self,
            init_qpos=np.array(
                [
                    1.18000162,
                    -1.91696992,
                    1.5561803,
                    -1.21203147,
                    -1.57465679,
                    -0.39695961,
                    0.0,
                ]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0
        return world_idx
