import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    convert_pose_repr,
    get_pose9_from_pose7,
)

# Observation keys of the policy, named as in UMI (`diffusion_policy/config/task/umi.yaml`).
# Note that `robot0_eef_rot_axis_angle` holds a 6D rotation, not an axis angle; the name is UMI's.
CAMERA_KEY = "camera0_rgb"
EEF_POS_KEY = "robot0_eef_pos"
EEF_ROT_KEY = "robot0_eef_rot_axis_angle"
GRIPPER_KEY = "robot0_gripper_width"

LOW_DIM_KEYS = (EEF_POS_KEY, EEF_ROT_KEY, GRIPPER_KEY)

# Action is UMI's 10D layout: position (3) + 6D rotation (6) + gripper (1)
ACTION_DIM = 10

# "delta" is accepted for the observation window but not advised: the window is anchored at its own last
# step, so entry k becomes the inverse of entry k+1 and the absolute reference is lost. UMI discourages it
# the same way, by comment rather than by check ("obs_pose_repr: relative # abs or rel" in
# config/task/umi.yaml), and its convert_pose_mat_rep accepts it too.


def get_shape_meta(obs_horizon, action_horizon, image_size):
    """
    Build the shape_meta that TimmObsEncoder and DiffusionUnetTimmPolicy consume.

    Only `shape`, `type` and `horizon` are read; UMI's `latency_steps` / `down_sample_steps` /
    `rotation_rep` are consumed by its sampler, which `UmiDpDataset` replaces.
    """
    return {
        "obs": {
            CAMERA_KEY: {
                "shape": [3, image_size[1], image_size[0]],
                "type": "rgb",
                "horizon": obs_horizon,
            },
            EEF_POS_KEY: {"shape": [3], "type": "low_dim", "horizon": obs_horizon},
            EEF_ROT_KEY: {"shape": [6], "type": "low_dim", "horizon": obs_horizon},
            GRIPPER_KEY: {"shape": [1], "type": "low_dim", "horizon": obs_horizon},
        },
        "action": {"shape": [ACTION_DIM], "horizon": action_horizon},
    }


class UmiDpDataset(DatasetBase):
    """
    Dataset to train the UMI diffusion policy.

    Unlike `DiffusionPolicyDataset`, the observation and action windows are separate sequences rather
    than a prefix and a suffix of one shared horizon:

        obs    = the `obs_horizon` steps ending at t (inclusive)
        action = the `action_horizon` steps starting at t

    Both are anchored to the measured end-effector pose at t, which is the last observation step and the
    only pose the rollout can observe when it predicts. Because the action chunk starts at the anchor,
    there is no index offset between what the policy emits and what gets executed.

    Data is returned unnormalized; `DiffusionUnetTimmPolicy` normalizes internally via its
    `LinearNormalizer`.
    """

    def setup_variables(self):
        skip = self.model_meta_info["data"]["skip"]
        obs_horizon = self.model_meta_info["data"]["obs_horizon"]

        # Index every timestep that has a full observation window behind it. Action windows are clipped
        # at the episode end instead of being dropped, so late-episode behavior stays represented.
        self.chunk_info_list = []
        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            for time_idx in range(obs_horizon - 1, episode_len):
                self.chunk_info_list.append((episode_idx, time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        obs_horizon = self.model_meta_info["data"]["obs_horizon"]
        action_horizon = self.model_meta_info["data"]["action_horizon"]
        image_size = self.model_meta_info["data"]["image_size"]
        camera_name = self.model_meta_info["data"]["camera_name"]
        episode_idx, time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(
            self.filenames[episode_idx], self.enable_rmb_cache, image_size=image_size
        ) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]

            obs_idxes = np.arange(time_idx - obs_horizon + 1, time_idx + 1)
            action_idxes = np.clip(
                np.arange(time_idx, time_idx + action_horizon), 0, episode_len - 1
            )

            measured_pose = get_pose9_from_pose7(
                rmb_data[DataKey.MEASURED_EEF_POSE][::skip][obs_idxes]
            )
            command_pose = get_pose9_from_pose7(
                rmb_data[DataKey.COMMAND_EEF_POSE][::skip][action_idxes]
            )
            gripper = rmb_data[DataKey.MEASURED_GRIPPER_JOINT_POS][::skip][obs_idxes]
            command_gripper = rmb_data[DataKey.COMMAND_GRIPPER_JOINT_POS][::skip][
                action_idxes
            ]
            images = rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][obs_idxes]

        # The anchor is the latest observation, so obs and action share one frame
        anchor_pose = measured_pose[-1]
        obs_pose = convert_pose_repr(
            measured_pose, anchor_pose, self.model_meta_info["state"]["pose_repr"]
        )
        action_pose = convert_pose_repr(
            command_pose, anchor_pose, self.model_meta_info["action"]["pose_repr"]
        )

        # Image augmentation lives in TimmObsEncoder, matching UMI, so only the dtype/layout
        # conversion happens here (`umi_dataset.py:266`)
        obs = {
            CAMERA_KEY: torch.tensor(
                np.moveaxis(images, -1, -3).astype(np.float32) / 255.0,
                dtype=torch.float32,
            ),
            EEF_POS_KEY: torch.tensor(obs_pose[:, :3], dtype=torch.float32),
            EEF_ROT_KEY: torch.tensor(obs_pose[:, 3:9], dtype=torch.float32),
            GRIPPER_KEY: torch.tensor(gripper, dtype=torch.float32),
        }
        action = torch.tensor(
            np.concatenate([action_pose, command_gripper], axis=1), dtype=torch.float32
        )

        return {"obs": obs, "action": action}
