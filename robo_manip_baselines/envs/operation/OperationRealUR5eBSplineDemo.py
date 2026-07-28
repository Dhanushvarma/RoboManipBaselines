import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0])
        self.duration = 0.5  # [s]


class OperationRealUR5eBSplineDemo:
    def __init__(
        self,
        robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        pointcloud_camera_ids=None,
        sanwa_keyboard_ids=None,
        dry_run=False,
        **env_kwargs,
    ):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        self.pointcloud_camera_ids = pointcloud_camera_ids
        self.sanwa_keyboard_ids = sanwa_keyboard_ids
        self.dry_run = dry_run
        # Extra kwargs (spline_rate, arm_frequency, max_joint_speed, ...) are
        # forwarded to the env, so they can be set from the config yaml.
        self.env_kwargs = env_kwargs
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealUR5eBSplineDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            pointcloud_camera_ids=self.pointcloud_camera_ids,
            sanwa_keyboard_ids=self.sanwa_keyboard_ids,
            dry_run=self.dry_run or getattr(self.args, "dry_run", False),
            **self.env_kwargs,
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
