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
    convert_pose_repr,
    get_pose9_from_pose7,
)

from .UmiDpDataset import (  # noqa: E402
    CAMERA_KEY,
    EEF_POS_KEY,
    EEF_ROT_KEY,
    GRIPPER_KEY,
)


class RolloutUmiDp(RolloutBase):
    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.state_pose_repr = self.model_meta_info["state"]["pose_repr"]
        self.action_pose_repr = self.model_meta_info["action"]["pose_repr"]

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
            f"  - camera: {self.model_meta_info['data']['camera_name']}, image size: {image_size}\n"
            f"  - backbone: {policy_meta_info['model_name']}"
        )

        self.load_ckpt()

    def setup_plot(self):
        fig_ax = plt.subplots(
            1, 2, figsize=(13.5, 6.0), dpi=60, squeeze=False, constrained_layout=True
        )
        super().setup_plot(fig_ax)

    def reset_variables(self):
        super().reset_variables()

        self.obs_buf = None
        self.policy_action_buf = None

    def get_obs(self):
        """Get the current observation in absolute form, as un-anchored numpy arrays."""
        camera_name = self.model_meta_info["data"]["camera_name"]
        image_size = self.model_meta_info["data"]["image_size"]

        image = self.info["rgb_images"][camera_name]
        image = cv2.resize(image, tuple(image_size))
        image = np.moveaxis(image, -1, -3).astype(np.float32) / 255.0

        return {
            CAMERA_KEY: image,
            "eef_pose": get_pose9_from_pose7(
                self.motion_manager.get_data(DataKey.MEASURED_EEF_POSE, self.obs)
            ),
            GRIPPER_KEY: self.motion_manager.get_data(
                DataKey.MEASURED_GRIPPER_JOINT_POS, self.obs
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
            CAMERA_KEY: np.stack([obs[CAMERA_KEY] for obs in self.obs_buf]),
            EEF_POS_KEY: eef_pose[:, :3],
            EEF_ROT_KEY: eef_pose[:, 3:9],
            GRIPPER_KEY: np.stack([obs[GRIPPER_KEY] for obs in self.obs_buf]),
        }
        input_data = {
            key: torch.tensor(value, dtype=torch.float32)[torch.newaxis].to(self.device)
            for key, value in input_data.items()
        }

        return input_data, anchor_pose

    def infer_policy(self):
        self.update_obs_buf()

        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            input_data, anchor_pose = self.get_policy_input()
            # UMI's predict_action returns the chunk unsliced and starting at the anchor step, so the
            # whole chunk is un-anchored here with no offset or prefix handling
            action = self.policy.predict_action(input_data)["action"][0]
            action = action.cpu().detach().numpy().astype(np.float64)
            action[:, :9] = convert_pose_repr(
                action[:, :9], anchor_pose, self.action_pose_repr, backward=True
            )
            self.policy_action_buf = list(action)

        self.policy_action = self.policy_action_buf.pop(0)
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def draw_plot(self):
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        camera_name = self.model_meta_info["data"]["camera_name"]
        self.ax[0, 0].imshow(self.info["rgb_images"][camera_name])
        self.ax[0, 0].set_title(camera_name, fontsize=20)
        self.plot_action(self.ax[0, 1])

        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )
