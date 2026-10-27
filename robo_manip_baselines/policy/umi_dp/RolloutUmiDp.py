import cv2
import matplotlib.pylab as plt
import numpy as np
import torch
import torchvision

from .UmiDpUtils import UMI_PATH  # noqa: F401  (must precede any diffusion_policy import)

from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder  # noqa: E402
from diffusion_policy.policy.diffusion_unet_timm_policy import (  # noqa: E402
    DiffusionUnetTimmPolicy,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402

from robo_manip_baselines.common import (  # noqa: E402
    DataKey,
    RolloutBase,
    convert_data_from_policy,
    convert_pose_repr,
    get_pose9_from_pose7,
)

from .UmiDpDataset import (  # noqa: E402
    EEF_POS_KEY,
    EEF_ROT_KEY,
    GRIPPER_KEY,
    get_camera_key,
    select_eef,
)
from .UmiDpEllipsoid import (  # noqa: E402
    ELLIPSOID_DIM,
    decode_ellipsoid,
    get_anchor_rotation,
    get_ellipsoid_command,
)

# Columns per end-effector of the action keys that hold every end-effector's data:
# (in the policy's action, in the env's command)
EEF_ACTION_DIMS = {
    DataKey.COMMAND_EEF_POSE: (9, 7),
    DataKey.COMMAND_GRIPPER_JOINT_POS: (1, 1),
}


class RolloutUmiDp(RolloutBase):
    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.state_pose_repr = self.model_meta_info["state"]["pose_repr"]
        self.action_pose_repr = self.model_meta_info["action"]["pose_repr"]
        # Checkpoints from before the ellipsoid have no entry
        self.ellipsoid_info = self.model_meta_info["action"].get("ellipsoid")
        # A whole-body controller env takes the ellipsoid as a command
        self.send_ellipsoid = (
            DataKey.COMMAND_EEF_ELLIPSOID in self.env.unwrapped.command_keys_for_step
        )
        if self.ellipsoid_info is not None and not self.send_ellipsoid:
            print(
                f"[{self.__class__.__name__}] The env takes no "
                f"{DataKey.COMMAND_EEF_ELLIPSOID}, so the predicted ellipsoid is not sent."
            )
        # End-effector the policy drives on an env with several (older checkpoints have none)
        self.eef_idx = self.model_meta_info["data"].get("eef_idx")
        num_eef = DataKey.get_num_eef(self.env)
        if self.eef_idx is None and num_eef > 1:
            raise ValueError(
                f"[{self.__class__.__name__}] The env has {num_eef} end-effectors, and "
                "the policy was trained on data with one (without --eef_idx)."
            )
        if self.eef_idx is not None and not 0 <= self.eef_idx < num_eef:
            raise ValueError(
                f"[{self.__class__.__name__}] The policy drives end-effector "
                f"{self.eef_idx}, and the env has {num_eef}."
            )

    def setup_policy(self):
        policy_meta_info = self.model_meta_info["policy"]
        shape_meta = policy_meta_info["shape_meta"]
        image_size = self.model_meta_info["data"]["image_size"]

        # Augmentation is training-only; only the resize is kept so the input size matches the backbone
        transforms = [
            torchvision.transforms.Resize(size=image_size[1], antialias=True),
        ]
        if policy_meta_info.get("imagenet_norm", False):
            transforms.append(
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )

        obs_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name=policy_meta_info["model_name"],
            pretrained=False,
            frozen=False,
            global_pool="",
            transforms=transforms,
            feature_aggregation=policy_meta_info["feature_aggregation"],
            downsample_ratio=32,
            position_encording="sinusoidal",
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=False,
        )

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        self.policy = DiffusionUnetTimmPolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            num_inference_steps=policy_meta_info["num_inference_steps"],
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs horizon: {self.model_meta_info['data']['obs_horizon']}, "
            f"action horizon: {self.model_meta_info['data']['action_horizon']}\n"
            f"  - image size: {image_size}\n"
            f"  - backbone: {policy_meta_info['model_name']}"
        )

        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            1,
            len(self.camera_names) + 1,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        super().setup_plot(fig_ax)

        if self.eef_idx is not None:
            # The env's scales cover every end-effector, the action one of them
            self.action_plot_scale = np.concatenate(
                [
                    select_eef(
                        DataKey.get_plot_scale_for_policy(key, self.env),
                        self.eef_idx,
                        EEF_ACTION_DIMS[key][0],
                    )
                    if key in EEF_ACTION_DIMS
                    else DataKey.get_plot_scale_for_policy(key, self.env)
                    for key in self.action_keys
                ]
            )

    def reset_variables(self):
        super().reset_variables()

        self.obs_buf = None
        self.policy_action_buf = None
        self.policy_ellipsoid_buf = None
        # Unit-determinant ellipsoid targets in world axes: the latest chunk, and this step's
        self.policy_ellipsoid_chunk = None
        self.policy_ellipsoid = None

    def get_obs(self):
        """Get the current observation in absolute form, as un-anchored numpy arrays."""
        image_size = self.model_meta_info["data"]["image_size"]

        images = {}
        for camera_idx, camera_name in enumerate(self.camera_names):
            image = self.info["rgb_images"][camera_name]
            image = cv2.resize(image, tuple(image_size))
            images[get_camera_key(camera_idx)] = (
                np.moveaxis(image, -1, -3).astype(np.float32) / 255.0
            )

        return {
            **images,
            "eef_pose": get_pose9_from_pose7(
                select_eef(
                    self.motion_manager.get_data(DataKey.MEASURED_EEF_POSE, self.obs),
                    self.eef_idx,
                    7,
                )
            ),
            GRIPPER_KEY: select_eef(
                self.motion_manager.get_data(
                    DataKey.MEASURED_GRIPPER_JOINT_POS, self.obs
                ),
                self.eef_idx,
                1,
            ),
        }

    def update_obs_buf(self):
        obs = self.get_obs()

        if self.obs_buf is None:
            self.obs_buf = [obs] * self.model_meta_info["data"]["obs_horizon"]
        else:
            self.obs_buf.pop(0)
            self.obs_buf.append(obs)

    def get_policy_input(self):
        """
        Build the policy input, anchoring the whole observation window to its latest step.

        The window shares one anchor, exactly as `UmiDpDataset` builds it. Anchoring each entry to its
        own step would erase the motion history the policy was trained on.
        """
        anchor_pose = self.obs_buf[-1]["eef_pose"]
        eef_pose = convert_pose_repr(
            np.stack([obs["eef_pose"] for obs in self.obs_buf]),
            anchor_pose,
            self.state_pose_repr,
        )

        input_data = {
            **{
                get_camera_key(camera_idx): np.stack(
                    [obs[get_camera_key(camera_idx)] for obs in self.obs_buf]
                )
                for camera_idx in range(len(self.camera_names))
            },
            EEF_POS_KEY: eef_pose[:, :3],
            EEF_ROT_KEY: eef_pose[:, 3:9],
            GRIPPER_KEY: np.stack([obs[GRIPPER_KEY] for obs in self.obs_buf]),
        }
        input_data = {
            key: torch.tensor(value, dtype=torch.float32)[torch.newaxis].to(self.device)
            for key, value in input_data.items()
        }

        return input_data, anchor_pose

    def split_ellipsoid(self, action, anchor_pose):
        """
        Remove the ellipsoid dims from a predicted chunk and keep them as world-axis targets.

        Returns the chunk the action keys describe, so everything downstream sees the width it expects. The
        targets go to `policy_ellipsoid_chunk`, and an env that takes them gets one per step (see
        set_ellipsoid_command_data).
        """
        if self.ellipsoid_info is None:
            return action
        self.policy_ellipsoid_chunk = decode_ellipsoid(
            action[:, -ELLIPSOID_DIM:], get_anchor_rotation(anchor_pose)
        )
        return action[:, :-ELLIPSOID_DIM]

    def infer_policy(self):
        self.update_obs_buf()

        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data, anchor_pose = self.get_policy_input()
            # UMI's predict_action returns the chunk unsliced and starting at the anchor step, so the
            # whole chunk is un-anchored here with no offset or prefix handling
            action = self.policy.predict_action(input_data)["action"][0]
            action = action.cpu().detach().numpy().astype(np.float64)
            action = self.split_ellipsoid(action, anchor_pose)
            action[:, :9] = convert_pose_repr(
                action[:, :9], anchor_pose, self.action_pose_repr, backward=True
            )
            self.policy_action_buf = list(action)
            if self.policy_ellipsoid_chunk is not None:
                self.policy_ellipsoid_buf = list(self.policy_ellipsoid_chunk)

        self.policy_action = self.policy_action_buf.pop(0)
        if self.policy_ellipsoid_buf:
            self.policy_ellipsoid = self.policy_ellipsoid_buf.pop(0)
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def set_command_data(self, action_keys=None):
        """As in RolloutBase, or for one end-effector of several (eef_idx); then the ellipsoid, if the env takes it."""
        if self.eef_idx is None:
            super().set_command_data(action_keys)
        else:
            self.set_one_eef_command_data(action_keys)
        if self.send_ellipsoid:
            self.set_ellipsoid_command_data()

    def set_one_eef_command_data(self, action_keys=None):
        """
        Set the commands on an env with several end-effectors.

        The action has one end-effector's pose and gripper, and the env takes every end-effector's
        command. The other end-effectors are sent their current command, so they hold.
        """
        if action_keys is None:
            action_keys = self.action_keys

        is_skip = self.rollout_time_idx % self.args.skip != 0
        action_idx = 0
        for key in action_keys:
            if key in EEF_ACTION_DIMS:
                action_dim, command_width = EEF_ACTION_DIMS[key]
                eef_command = convert_data_from_policy(
                    self.policy_action[action_idx : action_idx + action_dim], key
                )
                command = self.motion_manager.get_command_data(key)
                select_eef(command, self.eef_idx, command_width)[:] = eef_command
            else:
                action_dim = DataKey.get_dim_for_policy(key, self.env)
                command = convert_data_from_policy(
                    self.policy_action[action_idx : action_idx + action_dim], key
                )
            self.motion_manager.set_command_data(key, command, is_skip)
            action_idx += action_dim

    def set_ellipsoid_command_data(self):
        """
        Send this step's predicted shape as the driven end-effector's ellipsoid target.

        The other end-effectors get no target (NaN), and neither does any before the first
        prediction or with a policy that predicts none, so their controller task stays off.
        """
        command = np.full(
            DataKey.get_dim(DataKey.COMMAND_EEF_ELLIPSOID, self.env), np.nan
        )
        if self.policy_ellipsoid is not None:
            select_eef(command, self.eef_idx, ELLIPSOID_DIM)[:] = get_ellipsoid_command(
                self.policy_ellipsoid
            )
        is_skip = self.rollout_time_idx % self.args.skip != 0
        self.motion_manager.set_command_data(
            DataKey.COMMAND_EEF_ELLIPSOID, command, is_skip
        )

    def draw_plot(self):
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        self.plot_images(self.ax[0, 0 : len(self.camera_names)])
        self.plot_action(self.ax[0, len(self.camera_names)])

        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
