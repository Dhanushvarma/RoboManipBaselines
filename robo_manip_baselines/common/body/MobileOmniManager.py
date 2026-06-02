import dataclasses

import numpy as np

from ..data.DataKey import DataKey
from ..utils.MathUtils import get_se2_pose_from_se3, get_se3_from_se2_pose
from .BodyManagerBase import BodyConfigBase, BodyManagerBase


class MobileOmniManager(BodyManagerBase):
    """Manager for omni-directional mobile base.

    The base operates in the SE2 plane and exposes three independent command
    channels, any of which an env may drive (see MobileBaseUtils for the control
    modes that select between them):

      - velocity (``target_vel``): (vx, vy, vw) [m/s, m/s, rad/s].
      - absolute pose (``target_pos``): (x, y, yaw[rad]) in the odometry frame,
        whose origin is zeroed at each episode reset.
      - relative pose (``target_pos_rel``): (dx, dy, dyaw[rad]) delta.

    The relative pose mirrors the end-effector convention in ArmManager: the
    delta is expressed in the previous pose's frame (``prev^-1 * current``),
    computed with the same SE3 machinery embedded in the z=0 plane (MathUtils).
    Velocity carries no absolute/relative distinction (a rate has no fixed
    origin); its reference frame is handled at the env/hardware boundary.
    """

    SUPPORTED_DATA_KEYS = [
        DataKey.MEASURED_MOBILE_OMNI_VEL,
        DataKey.COMMAND_MOBILE_OMNI_VEL,
        DataKey.MEASURED_MOBILE_OMNI_POS,
        DataKey.COMMAND_MOBILE_OMNI_POS,
        DataKey.MEASURED_MOBILE_OMNI_POS_REL,
        DataKey.COMMAND_MOBILE_OMNI_POS_REL,
    ]

    def reset(self, init=False):
        self.target_vel = np.zeros(3)
        # Absolute target pose (x, y, yaw) in the odometry frame. Starts at the
        # origin, matching the odometry reset performed by the env on reset.
        self.target_pos = np.zeros(3)
        # Last applied relative command (dx, dy, dyaw) in the previous pose's
        # frame. Returned for the COMMAND_MOBILE_OMNI_POS_REL key.
        self.target_pos_rel = np.zeros(3)

    def set_command_data(self, key, command, is_skip=False):
        if key == DataKey.COMMAND_MOBILE_OMNI_VEL:
            self.set_command_vel(command)
        elif key == DataKey.COMMAND_MOBILE_OMNI_POS:
            self.set_command_pos(command)
        elif key == DataKey.COMMAND_MOBILE_OMNI_POS_REL:
            self.set_command_pos_rel(command, is_skip)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def set_command_vel(self, vel):
        self.target_vel = vel

    def set_command_pos(self, pos):
        pos = np.asarray(pos, dtype=np.float64)
        # Record the implied relative command so the REL command channel stays
        # consistent with the absolute target even in absolute mode.
        prev_se3 = get_se3_from_se2_pose(self.target_pos)
        new_se3 = get_se3_from_se2_pose(pos)
        self.target_pos_rel = get_se2_pose_from_se3(prev_se3.actInv(new_se3))
        self.target_pos = pos.copy()

    def set_command_pos_rel(self, pos_rel, is_skip=False):
        # Mirrors ArmManager.set_command_eef_pose_rel: on skip the target is
        # held (zero delta); otherwise the delta is composed onto the current
        # target in the target's own frame (right multiplication).
        pos_rel = np.zeros(3) if is_skip else np.asarray(pos_rel, dtype=np.float64)
        self.target_pos_rel = pos_rel.copy()
        new_se3 = get_se3_from_se2_pose(self.target_pos) * get_se3_from_se2_pose(pos_rel)
        self.target_pos = get_se2_pose_from_se3(new_se3)

    def get_command_data(self, key):
        if key == DataKey.COMMAND_MOBILE_OMNI_VEL:
            return self.get_command_vel()
        elif key == DataKey.COMMAND_MOBILE_OMNI_POS:
            return self.get_command_pos()
        elif key == DataKey.COMMAND_MOBILE_OMNI_POS_REL:
            return self.get_command_pos_rel()
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid command data key: {key}"
            )

    def get_command_vel(self):
        return self.target_vel

    def get_command_pos(self):
        return self.target_pos

    def get_command_pos_rel(self):
        return self.target_pos_rel

    def draw_markers(self):
        pass


@dataclasses.dataclass
class MobileOmniConfig(BodyConfigBase):
    """Configuration for omni-directional mobile base."""

    BodyManagerClass = MobileOmniManager
