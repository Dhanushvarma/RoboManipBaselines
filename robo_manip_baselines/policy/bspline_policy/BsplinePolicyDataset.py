import cv2
import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    convert_data_to_policy,
)

from . import BsplineAdapter as adapter


class BsplinePolicyDataset(DatasetBase):
    """Dataset to train B-spline policy.

    One sample is ``(observations ending at frame t, B-spline parameters
    anchored at t)``. Unlike an action-chunking dataset the target is not a
    horizon of actions but a ``(chunk_size + 2*degree, 1 + action_dim)`` matrix:
    column 0 holds the knot vector in frames relative to ``t``, the rest holds
    control points.

    Two departures from ``DiffusionPolicyDataset`` worth knowing about:

    * **Actions are fitted at full rate; only observations honour ``--skip``.**
      The adaptive fitter places knots by trajectory curvature, so subsampling
      shrinks the sample count without shrinking the knot count and the
      compression benefit evaporates (measured 5.41x -> 1.85x at ``--skip 3``).
      Observations are still strided by ``skip``, so ``n_obs_steps`` frames
      spaced ``skip`` apart end at ``t``.
    * **Action augmentation is disabled.** ``DatasetBase.augment_data`` adds
      Gaussian noise to the action tensor, which here would perturb the knot
      column and can destroy the non-decreasing property a valid B-spline
      requires.
    """

    def setup_variables(self):
        bspline_info = self.model_meta_info["data"]["bspline"]

        self.fitter = adapter.build_fitter(
            self.filenames,
            self.model_meta_info["action"]["keys"],
            bspline_info,
            cache_dir=bspline_info.get("cache_dir"),
            verbose=False,
        )

        skip = self.model_meta_info["data"]["skip"]
        n_obs_steps = self.model_meta_info["data"]["n_obs_steps"]

        # One sample per timestep that has a fitted chunk. Observations are
        # clamped at the episode start rather than dropped, which mirrors DP's
        # pad_before convention.
        self.chunk_info_list = []
        for episode_idx, episode_len in enumerate(self.fitter.episode_lengths):
            for time_idx in range(int(episode_len)):
                self.chunk_info_list.append((episode_idx, time_idx))

        self.obs_offsets = np.arange(-(n_obs_steps - 1), 1) * skip

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        image_size = self.model_meta_info["data"]["image_size"]
        episode_idx, time_idx = self.chunk_info_list[chunk_idx]
        episode_len = int(self.fitter.episode_lengths[episode_idx])

        time_idxes = np.clip(time_idx + self.obs_offsets, 0, episode_len - 1)

        with RmbData(
            self.filenames[episode_idx], self.enable_rmb_cache, image_size=image_size
        ) as rmb_data:
            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros((len(time_idxes), 0), dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        convert_data_to_policy(
                            np.asarray(rmb_data[key][:])[time_idxes], key
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load images. h5py fancy indexing requires strictly increasing
            # indices, but clamping the observation window at the episode start
            # produces duplicates (e.g. [0, 0] at t=0), so read the unique
            # frames once and expand back.
            if len(self.model_meta_info["image"]["camera_names"]) == 0:
                images = None
            else:
                unique_idxes, inverse = np.unique(time_idxes, return_inverse=True)
                images = np.stack(
                    [
                        np.asarray(
                            rmb_data[DataKey.get_rgb_image_key(camera_name)][
                                unique_idxes
                            ]
                        )[inverse]
                        for camera_name in self.model_meta_info["image"]["camera_names"]
                    ],
                    axis=0,
                )

        # The action target is the pre-fitted B-spline parameter matrix, not a
        # slice of the recorded action sequence.
        action = self.fitter.get_chunk(episode_idx, time_idx).astype(np.float64)

        if images is not None:
            images = self._resize_images(images, image_size)

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        if images is None:
            images_tensor = None
        else:
            images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Convert to data structure of policy input and output
        if len(self.model_meta_info["image"]["camera_names"]) == 0:
            data = {"obs": state_tensor, "action": action_tensor}
        else:
            data = {"obs": {}, "action": action_tensor}
            if len(self.model_meta_info["state"]["keys"]) > 0:
                data["obs"]["state"] = state_tensor
            for camera_idx, camera_name in enumerate(
                self.model_meta_info["image"]["camera_names"]
            ):
                data["obs"][DataKey.get_rgb_image_key(camera_name)] = images_tensor[
                    camera_idx
                ]

        return data

    @staticmethod
    def _resize_images(images, image_size):
        """Resize ``(n_camera, n_obs, H, W, C)`` images to ``(width, height)``.

        ``RmbData`` honours its ``image_size`` argument only for the Compact
        (``.rmb``) format, where images are decoded from mp4. For
        RmbData-SingleHDF5 it hands back the raw h5py dataset at whatever
        resolution was recorded, so the resize has to happen here or a
        SingleHDF5 dataset could only ever be trained at its native resolution.
        """
        target_w, target_h = int(image_size[0]), int(image_size[1])
        current_h, current_w = images.shape[2:4]
        if (current_w, current_h) == (target_w, target_h):
            return images

        resized = np.empty(
            images.shape[:2] + (target_h, target_w) + images.shape[4:],
            dtype=images.dtype,
        )
        for camera_idx in range(images.shape[0]):
            for obs_idx in range(images.shape[1]):
                resized[camera_idx, obs_idx] = cv2.resize(
                    images[camera_idx, obs_idx],
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA,
                )
        return resized

    def augment_data(self, state, action, images):
        # Deliberately pass action=None: noise on the knot column can break the
        # non-decreasing property a valid B-spline requires, and the paper's
        # augmentation story is on images, not on spline parameters.
        state, _, images = super().augment_data(state, None, images)

        if images is not None:
            # Adjust to a range from -1 to 1 to match the original implementation
            images = images * 2.0 - 1.0

        return state, action, images
