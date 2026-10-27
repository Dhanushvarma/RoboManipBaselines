import dataclasses
import os
import pickle
import tempfile
import types
from typing import Any

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robo_manip_baselines.common import (
    BodyConfigBase,
    BodyManagerBase,
    DataKey,
    MotionManager,
    get_pose9_from_pose7,
)
from robo_manip_baselines.policy.umi_dp import RolloutUmiDp, UmiDpDataset

ACTION_KEYS = [DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS]


def get_random_pose7(rng, num_frames):
    quat = Rotation.random(num_frames, random_state=rng).as_quat()
    return np.concatenate([rng.normal(size=(num_frames, 3)), quat[:, [3, 0, 1, 2]]], 1)


def write_episode(path, eef_pose_list, gripper_list):
    """Write the keys UmiDp reads, for the given end-effectors; commands lead by one frame."""
    eef_pose = np.concatenate(eef_pose_list, axis=1)
    gripper = np.concatenate(gripper_list, axis=1)
    with h5py.File(path, "w") as f:
        f[DataKey.TIME] = 0.1 * np.arange(len(eef_pose))
        f[DataKey.MEASURED_EEF_POSE] = eef_pose
        f[DataKey.COMMAND_EEF_POSE] = np.roll(eef_pose, -1, axis=0)
        f[DataKey.MEASURED_GRIPPER_JOINT_POS] = gripper
        f[DataKey.COMMAND_GRIPPER_JOINT_POS] = np.roll(gripper, -1, axis=0)


def make_model_meta_info(eef_idx):
    return {
        "data": {
            "skip": 3,
            "obs_horizon": 2,
            "action_horizon": 16,
            "image_size": [224, 224],
            "sample_every_frame": True,
            "eef_idx": eef_idx,
        },
        "image": {
            "camera_names": [],
            "aug_erasing_scale": 0.0,
            "aug_color_scale": 0.0,
            "aug_affine_scale": 0.0,
            "aug_std": 0.0,
        },
        "state": {"pose_repr": "relative"},
        "action": {"keys": ACTION_KEYS, "pose_repr": "relative"},
    }


def make_two_arm_episodes(tmp_dir):
    """Write a two-arm episode and a one-arm episode holding the second arm's data."""
    rng = np.random.default_rng(0)
    num_frames = 40
    eef_pose_list = [get_random_pose7(rng, num_frames) for _ in range(2)]
    gripper_list = [rng.uniform(size=(num_frames, 1)) for _ in range(2)]
    two_arm_path = os.path.join(tmp_dir, "two_arm.hdf5")
    one_arm_path = os.path.join(tmp_dir, "one_arm.hdf5")
    write_episode(two_arm_path, eef_pose_list, gripper_list)
    write_episode(one_arm_path, eef_pose_list[1:], gripper_list[1:])
    return two_arm_path, one_arm_path


def test_dataset_learns_one_eef():
    with tempfile.TemporaryDirectory() as tmp_dir:
        two_arm_path, one_arm_path = make_two_arm_episodes(tmp_dir)
        two_arm = UmiDpDataset([two_arm_path], make_model_meta_info(eef_idx=1))
        one_arm = UmiDpDataset([one_arm_path], make_model_meta_info(eef_idx=None))

        assert len(two_arm) == len(one_arm) > 0
        for chunk_idx in range(len(one_arm)):
            expected = one_arm[chunk_idx]
            data = two_arm[chunk_idx]
            assert data["obs"].keys() == expected["obs"].keys()
            for key, value in expected["obs"].items():
                assert torch.equal(data["obs"][key], value), key
            assert torch.equal(data["action"], expected["action"])


def test_dataset_needs_a_valid_eef_idx():
    with tempfile.TemporaryDirectory() as tmp_dir:
        two_arm_path, _ = make_two_arm_episodes(tmp_dir)
        for eef_idx, message in (
            (None, "--eef_idx"),
            (2, "no eef_idx 2"),
            (-1, "no eef_idx -1"),
        ):
            try:
                UmiDpDataset([two_arm_path], make_model_meta_info(eef_idx))
            except ValueError as error:
                assert message in str(error), error
            else:
                raise AssertionError(f"eef_idx {eef_idx} was accepted")


class FakeArmManager(BodyManagerBase):
    """An arm with an end-effector and a gripper joint, which keeps its command as given."""

    SUPPORTED_DATA_KEYS = ACTION_KEYS

    def __init__(self, env, body_config):
        super().__init__(env, body_config)
        eef_idx = body_config.eef_idx
        self.eef_pose = np.array([eef_idx, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.gripper_joint_pos = np.array([10.0 + eef_idx])

    def get_eef_pose_from_joint_pos(self, arm_joint_pos):
        return np.array([arm_joint_pos[0], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def set_command_data(self, key, command, is_skip=False):
        if key == DataKey.COMMAND_EEF_POSE:
            eef_idx = self.body_config.eef_idx
            self.eef_pose = command[7 * eef_idx : 7 * (eef_idx + 1)].copy()
        else:
            self.gripper_joint_pos = command[
                self.body_config.gripper_joint_idxes_in_gripper_joint_pos
            ].copy()

    def get_command_data(self, key):
        if key == DataKey.COMMAND_EEF_POSE:
            return self.eef_pose.copy()
        return self.gripper_joint_pos.copy()


@dataclasses.dataclass
class FakeArmConfig(BodyConfigBase):
    BodyManagerClass = FakeArmManager

    HAS_ARM = True
    HAS_GRIPPER = True
    HAS_EEF = True

    arm_joint_idxes: Any
    gripper_joint_idxes: Any
    gripper_joint_idxes_in_gripper_joint_pos: Any
    eef_idx: Any


def make_rollout(eef_idx, num_arms=2):
    """A RolloutUmiDp on an env with num_arms arms, set up only as far as observing and commanding."""
    env = types.SimpleNamespace(
        body_config_list=[
            FakeArmConfig(
                arm_joint_idxes=np.arange(4 * arm_idx, 4 * arm_idx + 3),
                gripper_joint_idxes=np.array([4 * arm_idx + 3]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([arm_idx]),
                eef_idx=arm_idx,
            )
            for arm_idx in range(num_arms)
        ],
        get_joint_pos_from_obs=lambda obs: obs["joint_pos"],
        get_gripper_joint_pos_from_obs=lambda obs: obs["joint_pos"][3::4],
        command_keys_for_step=ACTION_KEYS,
    )
    env.unwrapped = env

    rollout = RolloutUmiDp.__new__(RolloutUmiDp)
    rollout.env = env
    rollout.motion_manager = MotionManager(env)
    rollout.args = types.SimpleNamespace(skip=3)
    rollout.model_meta_info = make_model_meta_info(eef_idx)
    rollout.action_keys = ACTION_KEYS
    rollout.camera_names = []
    rollout.eef_idx = eef_idx
    rollout.send_ellipsoid = False
    rollout.rollout_time_idx = 0
    return rollout


def test_rollout_takes_eef_idx_from_the_checkpoint():
    with tempfile.TemporaryDirectory() as tmp_dir:
        for eef_idx, num_arms, message in (
            (1, 2, None),
            (None, 1, None),
            (None, 2, "without --eef_idx"),
            (1, 1, "drives end-effector 1"),
        ):
            model_meta_info = make_model_meta_info(eef_idx)
            if eef_idx is None:
                # As in checkpoints from before the option
                del model_meta_info["data"]["eef_idx"]
            model_meta_info["state"].update(keys=[], example=np.zeros(10))
            model_meta_info["action"]["example"] = np.zeros(10)
            with open(os.path.join(tmp_dir, "model_meta_info.pkl"), "wb") as f:
                pickle.dump(model_meta_info, f)

            rollout = make_rollout(eef_idx=None, num_arms=num_arms)
            rollout.args = types.SimpleNamespace(
                checkpoint=os.path.join(tmp_dir, "policy_last.ckpt"),
                skip=None,
                skip_draw=None,
            )
            try:
                rollout.setup_model_meta_info()
            except ValueError as error:
                assert message is not None and message in str(error), error
            else:
                assert message is None, f"eef_idx {eef_idx} with {num_arms} arms"
                assert rollout.eef_idx == eef_idx


def test_rollout_observes_one_eef():
    rollout = make_rollout(eef_idx=1)
    rollout.obs = {"joint_pos": np.arange(8.0)}
    rollout.info = {"rgb_images": {}}

    obs = rollout.get_obs()
    assert np.array_equal(
        obs["eef_pose"], get_pose9_from_pose7([4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    )
    assert np.array_equal(obs["robot0_gripper_width"], [7.0])


def test_rollout_holds_the_other_eef():
    rollout = make_rollout(eef_idx=1)
    held_eef_pose = rollout.motion_manager.get_command_data(DataKey.COMMAND_EEF_POSE)
    quat = np.array([0.9, 0.1, 0.3, -0.2]) / np.linalg.norm([0.9, 0.1, 0.3, -0.2])
    eef_pose = np.concatenate([[0.1, 0.2, 0.3], quat])
    rollout.policy_action = np.concatenate([get_pose9_from_pose7(eef_pose), [0.5]])

    rollout.set_command_data()

    command_eef_pose = rollout.motion_manager.get_command_data(DataKey.COMMAND_EEF_POSE)
    assert np.array_equal(command_eef_pose[:7], held_eef_pose[:7])
    np.testing.assert_allclose(command_eef_pose[7:], eef_pose, atol=1e-12)
    command_gripper = rollout.motion_manager.get_command_data(
        DataKey.COMMAND_GRIPPER_JOINT_POS
    )
    assert np.array_equal(command_gripper, [10.0, 0.5])


if __name__ == "__main__":
    test_dataset_learns_one_eef()
    test_dataset_needs_a_valid_eef_idx()
    test_rollout_takes_eef_idx_from_the_checkpoint()
    test_rollout_observes_one_eef()
    test_rollout_holds_the_other_eef()
