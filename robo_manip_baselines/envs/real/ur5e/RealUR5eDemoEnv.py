import numpy as np

from .RealUR5eEnvBase import RealUR5eEnvBase


class RealUR5eDemoEnv(RealUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealUR5eEnvBase.__init__(
            self,
            init_qpos=np.array(
                [
                    +3.10825300,
                    -1.35423905,
                    +0.96194107,
                    -1.16364892,
                    -1.53427154,
                    -0.06221611,
                    +0.00000000,  # gripper
                ]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
