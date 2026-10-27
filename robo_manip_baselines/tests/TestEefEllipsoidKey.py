import dataclasses
import types
from typing import Any

import numpy as np

from robo_manip_baselines.common import (
    BodyConfigBase,
    BodyManagerBase,
    DataKey,
    MotionManager,
)


class FakeEllipsoidManager(BodyManagerBase):
    """Keeps the ellipsoid command of its end-effectors as given."""

    SUPPORTED_DATA_KEYS = [DataKey.COMMAND_EEF_ELLIPSOID]

    def __init__(self, env, body_config):
        super().__init__(env, body_config)
        self.eef_idxes = DataKey.get_eef_idxes(body_config)
        self.ellipsoid = np.full(6 * len(self.eef_idxes), np.nan)

    def set_command_data(self, key, command, is_skip=False):
        self.ellipsoid = np.concatenate(
            [command[6 * eef_idx : 6 * (eef_idx + 1)] for eef_idx in self.eef_idxes]
        )

    def get_command_data(self, key):
        return self.ellipsoid.copy()


@dataclasses.dataclass
class FakeEllipsoidConfig(BodyConfigBase):
    BodyManagerClass = FakeEllipsoidManager

    HAS_EEF = True

    eef_idx: Any


class FakePoseOnlyManager(BodyManagerBase):
    """An end-effector body that takes no ellipsoid."""

    SUPPORTED_DATA_KEYS = [DataKey.COMMAND_EEF_POSE]


@dataclasses.dataclass
class FakePoseOnlyConfig(BodyConfigBase):
    BodyManagerClass = FakePoseOnlyManager

    HAS_EEF = True

    eef_idx: Any


def make_env(body_config_list):
    env = types.SimpleNamespace(body_config_list=body_config_list)
    env.unwrapped = env
    return env


def get_shapes(num_eef):
    """Distinct 6D entries per end-effector, with the end-effector index in the first."""
    return np.concatenate(
        [[eef_idx, 1.0, 1.0, 0.1, 0.2, 0.3] for eef_idx in range(num_eef)]
    ).astype(np.float64)


def check_round_trip(env):
    num_eef = DataKey.get_num_eef(env)
    assert DataKey.get_dim(DataKey.COMMAND_EEF_ELLIPSOID, env) == 6 * num_eef
    assert DataKey.get_dim_for_policy(DataKey.COMMAND_EEF_ELLIPSOID, env) == 6 * num_eef
    assert np.array_equal(
        DataKey.get_plot_scale_for_policy(DataKey.COMMAND_EEF_ELLIPSOID, env),
        np.ones(6 * num_eef),
    )

    motion_manager = MotionManager(env)
    # No target before any is set
    assert np.isnan(
        motion_manager.get_command_data(DataKey.COMMAND_EEF_ELLIPSOID)
    ).all()

    shapes = get_shapes(num_eef)
    motion_manager.set_command_data(DataKey.COMMAND_EEF_ELLIPSOID, shapes)
    command = motion_manager.get_command_data(DataKey.COMMAND_EEF_ELLIPSOID)
    assert np.array_equal(command, shapes), command


def test_one_eef_per_body():
    check_round_trip(
        make_env([FakeEllipsoidConfig(eef_idx=0), FakeEllipsoidConfig(eef_idx=1)])
    )


def test_two_eef_body():
    check_round_trip(make_env([FakeEllipsoidConfig(eef_idx=[0, 1])]))


def test_two_eef_body_in_reverse_order():
    check_round_trip(make_env([FakeEllipsoidConfig(eef_idx=[1, 0])]))


def test_eef_without_ellipsoid_has_no_target():
    env = make_env([FakePoseOnlyConfig(eef_idx=0), FakeEllipsoidConfig(eef_idx=1)])
    motion_manager = MotionManager(env)
    shapes = get_shapes(2)
    motion_manager.set_command_data(DataKey.COMMAND_EEF_ELLIPSOID, shapes)

    command = motion_manager.get_command_data(DataKey.COMMAND_EEF_ELLIPSOID)
    assert np.isnan(command[0:6]).all(), command
    assert np.array_equal(command[6:12], shapes[6:12]), command


if __name__ == "__main__":
    test_one_eef_per_body()
    test_two_eef_body()
    test_two_eef_body_in_reverse_order()
    test_eef_without_ellipsoid_has_no_target()
