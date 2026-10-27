import numpy as np

from ..data.DataKey import DataKey


class MotionManager:
    """Manager for robot motion."""

    def __init__(self, env):
        self.env = env

        self.body_manager_list = []
        for body_config in self.env.unwrapped.body_config_list:
            self.body_manager_list.append(
                body_config.BodyManagerClass(self.env, body_config)
            )

    def reset(self):
        """Reset internal states."""

        for body_manager in self.body_manager_list:
            body_manager.reset()

    def set_command_data(self, key, command, is_skip=False):
        """Sets command data of the specified key."""

        is_handled = False
        for body_manager in self.body_manager_list:
            if key not in body_manager.SUPPORTED_DATA_KEYS:
                continue

            is_handled = True
            body_manager.set_command_data(key, command, is_skip)

        if not is_handled:
            raise ValueError(
                f"[{self.__class__.__name__}] Command data key is not supported by any body manager: {key}"
            )

    def get_data(self, key, obs=None):
        """
        Get data of the specified key.

        This is a wrapper to get command data and measured data from a unified method.
        """
        if key in DataKey.MEASURED_DATA_KEYS:
            return self.get_measured_data(key, obs)
        elif key in DataKey.COMMAND_DATA_KEYS:
            return self.get_command_data(key)
        else:
            raise ValueError(f"[{self.__class__.__name__}] Invalid data key: {key}")

    def get_measured_data(self, key, obs):
        """Get measured data of the specified key from observation."""
        if key == DataKey.MEASURED_JOINT_POS:
            return self.env.unwrapped.get_joint_pos_from_obs(obs)
        elif key == DataKey.MEASURED_JOINT_VEL:
            return self.env.unwrapped.get_joint_vel_from_obs(obs)
        elif key == DataKey.MEASURED_GRIPPER_JOINT_POS:
            return self.env.unwrapped.get_gripper_joint_pos_from_obs(obs)
        elif key == DataKey.MEASURED_EEF_POSE:
            measured_eef_pose_list = [None] * DataKey.get_num_eef(self.env)

            measured_joint_pos = self.env.unwrapped.get_joint_pos_from_obs(obs)

            for body_manager in self.body_manager_list:
                body_config = body_manager.body_config
                eef_idxes = DataKey.get_eef_idxes(body_config)
                if len(eef_idxes) == 0:
                    continue

                # Several end-effectors of one body come back concatenated
                eef_pose = body_manager.get_eef_pose_from_joint_pos(
                    measured_joint_pos[body_config.arm_joint_idxes]
                )
                for i, eef_idx in enumerate(eef_idxes):
                    measured_eef_pose_list[eef_idx] = eef_pose[7 * i : 7 * (i + 1)]

            return np.concatenate(measured_eef_pose_list)
        elif key == DataKey.MEASURED_EEF_WRENCH:
            return self.env.unwrapped.get_eef_wrench_from_obs(obs)
        elif key == DataKey.MEASURED_MOBILE_OMNI_POS:
            return self.env.unwrapped.get_mobile_pos_from_obs(obs)
        elif key == DataKey.MEASURED_MOBILE_OMNI_VEL:
            return self.env.unwrapped.get_mobile_vel_from_obs(obs)
        elif key == DataKey.MEASURED_WBC_MODE:
            return self.env.unwrapped.get_wbc_mode_from_obs(obs)
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid measured data key: {key}"
            )

    def get_command_data(self, key):
        """Get command data of the specified key."""
        supported_data_keys = [
            DataKey.COMMAND_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
            DataKey.COMMAND_EEF_POSE,
            DataKey.COMMAND_MOBILE_OMNI_VEL,
            DataKey.COMMAND_WBC_MODE,
            DataKey.COMMAND_EEF_ELLIPSOID,
        ]
        if key not in supported_data_keys:
            raise ValueError(
                f"[{self.__class__.__name__}] Command data key is not supported: {key}"
            )

        if key == DataKey.COMMAND_EEF_ELLIPSOID:
            # An end-effector that no manager gives a shape has no target
            command = np.full(DataKey.get_dim(key, self.env), np.nan)
        else:
            command = np.zeros(DataKey.get_dim(key, self.env))

        for body_manager in self.body_manager_list:
            if key not in body_manager.SUPPORTED_DATA_KEYS:
                continue

            single_command = body_manager.get_command_data(key)

            if key == DataKey.COMMAND_JOINT_POS:
                command[body_manager.body_config.arm_joint_idxes] = single_command[0]
                command[body_manager.body_config.gripper_joint_idxes] = single_command[
                    1
                ]
            elif key == DataKey.COMMAND_GRIPPER_JOINT_POS:
                command[
                    body_manager.body_config.gripper_joint_idxes_in_gripper_joint_pos
                ] = single_command
            elif key in (DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_EEF_ELLIPSOID):
                # One slot per end-effector: a pose7, or a 6D ellipsoid shape
                width = 7 if key == DataKey.COMMAND_EEF_POSE else 6
                for i, eef_idx in enumerate(
                    DataKey.get_eef_idxes(body_manager.body_config)
                ):
                    command[width * eef_idx : width * (eef_idx + 1)] = single_command[
                        width * i : width * (i + 1)
                    ]
            else:
                command[:] = single_command

        return command

    def draw_markers(self):
        """Draw markers of the current states."""

        for body_manager in self.body_manager_list:
            body_manager.draw_markers()
