import dataclasses
import types
from typing import Any

import numpy as np

from robo_manip_baselines.common import (
    BodyConfigBase,
    BodyManagerBase,
    DataKey,
    MobileOmniConfig,
    MotionManager,
)


class FakeEefManager(BodyManagerBase):
    """Reports one pose per end-effector, with x set to that end-effector's index."""

    SUPPORTED_DATA_KEYS = [DataKey.MEASURED_EEF_POSE, DataKey.COMMAND_EEF_POSE]

    def get_poses(self, offset):
        return np.concatenate(
            [
                [eef_idx + offset, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                for eef_idx in DataKey.get_eef_idxes(self.body_config)
            ]
        )

    def get_eef_pose_from_joint_pos(self, arm_joint_pos):
        return self.get_poses(0.0)

    def get_command_data(self, key):
        return self.get_poses(0.5)


@dataclasses.dataclass
class FakeEefConfig(BodyConfigBase):
    BodyManagerClass = FakeEefManager

    HAS_ARM = True
    HAS_EEF = True

    arm_joint_idxes: Any
    eef_idx: Any


def make_env(body_config_list):
    env = types.SimpleNamespace(
        body_config_list=body_config_list,
        get_joint_pos_from_obs=lambda obs: obs["joint_pos"],
    )
    env.unwrapped = env
    return env


def check_eef_slots(env, expected_x):
    motion_manager = MotionManager(env)
    obs = {"joint_pos": np.zeros(DataKey.get_dim(DataKey.MEASURED_JOINT_POS, env))}
    measured = motion_manager.get_measured_data(DataKey.MEASURED_EEF_POSE, obs)
    command = motion_manager.get_command_data(DataKey.COMMAND_EEF_POSE)

    assert DataKey.get_num_eef(env) == len(expected_x)
    assert DataKey.get_dim(DataKey.MEASURED_EEF_POSE, env) == 7 * len(expected_x)
    assert DataKey.get_dim_for_policy(DataKey.MEASURED_EEF_POSE, env) == 9 * len(
        expected_x
    )
    assert np.array_equal(measured[0::7], expected_x), measured
    assert np.array_equal(command[0::7], np.add(expected_x, 0.5)), command


def test_one_eef_per_body():
    env = make_env(
        [
            FakeEefConfig(arm_joint_idxes=np.arange(0, 7), eef_idx=0),
            FakeEefConfig(arm_joint_idxes=np.arange(7, 14), eef_idx=1),
            MobileOmniConfig(),
        ]
    )
    check_eef_slots(env, [0.0, 1.0])


def test_two_eef_body():
    env = make_env(
        [
            FakeEefConfig(arm_joint_idxes=np.arange(14), eef_idx=[0, 1]),
            MobileOmniConfig(),
        ]
    )
    check_eef_slots(env, [0.0, 1.0])


def test_two_eef_body_in_reverse_order():
    env = make_env([FakeEefConfig(arm_joint_idxes=np.arange(14), eef_idx=[1, 0])])
    check_eef_slots(env, [0.0, 1.0])


def test_two_eef_body_next_to_one_eef_body():
    env = make_env(
        [
            FakeEefConfig(arm_joint_idxes=np.arange(14), eef_idx=[0, 2]),
            FakeEefConfig(arm_joint_idxes=np.arange(14, 20), eef_idx=1),
        ]
    )
    check_eef_slots(env, [0.0, 1.0, 2.0])


if __name__ == "__main__":
    test_one_eef_per_body()
    test_two_eef_body()
    test_two_eef_body_in_reverse_order()
    test_two_eef_body_next_to_one_eef_body()
