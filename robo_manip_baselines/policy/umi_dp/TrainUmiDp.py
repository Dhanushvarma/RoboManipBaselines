import argparse
import copy

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from .UmiDpUtils import UMI_PATH  # noqa: F401  (must precede any diffusion_policy import)

from diffusion_policy.common.normalize_util import (  # noqa: E402
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to  # noqa: E402
from diffusion_policy.model.common.lr_scheduler import get_scheduler  # noqa: E402
from diffusion_policy.model.common.normalizer import LinearNormalizer  # noqa: E402
from diffusion_policy.model.diffusion.ema_model import EMAModel  # noqa: E402
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder  # noqa: E402
from diffusion_policy.policy.diffusion_unet_timm_policy import (  # noqa: E402
    DiffusionUnetTimmPolicy,
)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402

from robo_manip_baselines.common import (  # noqa: E402
    POSE_REPR_LIST,
    DataKey,
    RmbData,
    TrainBase,
)

from .UmiDpDataset import (  # noqa: E402
    ACTION_DIM,
    CAMERA_KEY,
    EEF_ROT_KEY,
    LOW_DIM_KEYS,
    UmiDpDataset,
    get_shape_meta,
)


class TrainUmiDp(TrainBase):
    """Train UMI's diffusion policy on RMB data (single arm)."""

    DatasetClass = UmiDpDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        # Defaults follow UMI's train_diffusion_unet_timm_umi_workspace.yaml and task/umi.yaml
        parser.set_defaults(batch_size=64)
        parser.set_defaults(num_epochs=120)
        parser.set_defaults(lr=3e-4)
        parser.set_defaults(skip=3)  # UMI's obs_down_sample_steps

        # State and action content are fixed by UMI's observation contract
        parser.set_defaults(
            state_keys=[DataKey.MEASURED_EEF_POSE, DataKey.MEASURED_GRIPPER_JOINT_POS]
        )
        parser.set_defaults(
            action_keys=[DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS]
        )

        # UMI's observation contract has exactly one camera, mapped to camera0_rgb
        parser.set_defaults(camera_names=["hand"])

        parser.add_argument(
            "--obs_horizon", type=int, default=2, help="number of observation steps"
        )
        parser.add_argument(
            "--action_horizon", type=int, default=16, help="number of predicted actions"
        )
        parser.add_argument(
            "--image_size",
            type=int,
            nargs=2,
            default=[224, 224],
            help="image size (width, height); the CLIP ViT backbone expects 224x224",
        )

        parser.add_argument(
            "--action_pose_repr",
            type=str,
            default="relative",
            choices=POSE_REPR_LIST,
            help="representation of the end-effector pose action (see convert_pose_repr)",
        )
        parser.add_argument(
            "--state_pose_repr",
            type=str,
            default="relative",
            choices=POSE_REPR_LIST,
            help="representation of the end-effector pose observation (see convert_pose_repr). "
            "'delta' is accepted but not advised; UMI offers only abs or relative here.",
        )

        parser.add_argument(
            "--model_name",
            type=str,
            default="vit_base_patch16_clip_224.openai",
            help="timm backbone name",
        )
        parser.add_argument(
            "--feature_aggregation",
            type=str,
            default="attention_pool_2d",
            help="how per-patch features are pooled in TimmObsEncoder",
        )
        parser.add_argument(
            "--frozen_backbone",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="whether to freeze the pretrained vision backbone",
        )
        parser.add_argument(
            "--imagenet_norm",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="whether to apply ImageNet normalization to images. Off matches UMI, whose "
            "TimmObsEncoder accepts but never applies its imagenet_norm flag.",
        )

        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="enable or disable exponential moving average (EMA)",
        )
        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
        )
        parser.add_argument(
            "--num_inference_steps",
            type=int,
            default=16,
            help="number of denoising steps at inference",
        )

    def setup_args(self):
        super().setup_args()

        if len(self.args.camera_names) != 1:
            raise ValueError(
                f"[{self.__class__.__name__}] UmiDp takes exactly one camera: "
                f"{self.args.camera_names}"
            )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"].update(
            {
                "camera_name": self.args.camera_names[0],
                "obs_horizon": self.args.obs_horizon,
                "action_horizon": self.args.action_horizon,
                "image_size": self.args.image_size,
            }
        )
        self.model_meta_info["state"]["pose_repr"] = self.args.state_pose_repr
        self.model_meta_info["action"]["pose_repr"] = self.args.action_pose_repr
        self.model_meta_info["policy"].update(
            {
                "use_ema": self.args.use_ema,
                "model_name": self.args.model_name,
                "feature_aggregation": self.args.feature_aggregation,
                "frozen_backbone": self.args.frozen_backbone,
                "imagenet_norm": self.args.imagenet_norm,
                "num_inference_steps": self.args.num_inference_steps,
            }
        )
        self.model_meta_info["policy"]["shape_meta"] = get_shape_meta(
            self.args.obs_horizon, self.args.action_horizon, self.args.image_size
        )

    def set_data_stats(self):
        """
        Record only what the base class plumbing needs.

        Input scaling is handled by the policy's `LinearNormalizer` (fitted in `setup_policy`), so the
        mean/std/range statistics that `TrainBase.set_data_stats` computes are unused here.
        """
        episode_len_list = []
        for filename in self.all_filenames:
            with RmbData(filename) as rmb_data:
                episode_len_list.append(
                    rmb_data[DataKey.TIME][:: self.args.skip].shape[0]
                )

        self.model_meta_info["state"]["example"] = np.zeros(ACTION_DIM)
        self.model_meta_info["action"]["example"] = np.zeros(ACTION_DIM)
        self.model_meta_info["data"].update(
            {
                "mean_episode_len": np.mean(episode_len_list),
                "min_episode_len": np.min(episode_len_list),
                "max_episode_len": np.max(episode_len_list),
            }
        )

    def fit_normalizer(self, dataset):
        """
        Fit UMI's LinearNormalizer by iterating the dataset (`umi_dataset.py::get_normalizer`).

        Statistics come from the transformed data the policy actually sees, so a non-absolute pose
        representation is handled without any separate reconstruction of the distribution. Rotations
        get an identity normalizer because 6D rotations are already unit-scale.
        """
        normalizer = LinearNormalizer()

        data_cache = {key: [] for key in (*LOW_DIM_KEYS, "action")}
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, num_workers=self.args.num_workers
        )
        for batch in tqdm(dataloader, desc="Iterating dataset to fit normalizer"):
            for key in LOW_DIM_KEYS:
                data_cache[key].append(batch["obs"][key].numpy())
            data_cache["action"].append(batch["action"].numpy())

        for key in data_cache:
            stacked = np.concatenate(data_cache[key])
            data_cache[key] = stacked.reshape(-1, stacked.shape[-1])

        action = data_cache["action"]
        normalizer["action"] = concatenate_normalizer(
            [
                get_range_normalizer_from_stat(array_to_stats(action[..., :3])),
                get_identity_normalizer_from_stat(array_to_stats(action[..., 3:9])),
                get_range_normalizer_from_stat(array_to_stats(action[..., 9:10])),
            ]
        )
        for key in LOW_DIM_KEYS:
            stat = array_to_stats(data_cache[key])
            if key == EEF_ROT_KEY:
                normalizer[key] = get_identity_normalizer_from_stat(stat)
            else:
                normalizer[key] = get_range_normalizer_from_stat(stat)
        normalizer[CAMERA_KEY] = get_image_identity_normalizer()

        return normalizer

    def setup_policy(self):
        shape_meta = self.model_meta_info["policy"]["shape_meta"]

        # UMI applies image augmentation inside the encoder rather than in the dataset
        transforms = [
            torchvision.transforms.RandomCrop(
                size=int(self.args.image_size[1] * 0.95)
            ),
            torchvision.transforms.Resize(
                size=self.args.image_size[1], antialias=True
            ),
            torchvision.transforms.ColorJitter(
                brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08
            ),
        ]
        if self.args.imagenet_norm:
            transforms.append(
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            )

        obs_encoder = TimmObsEncoder(
            shape_meta=shape_meta,
            model_name=self.args.model_name,
            pretrained=True,
            frozen=self.args.frozen_backbone,
            global_pool="",
            transforms=transforms,
            feature_aggregation=self.args.feature_aggregation,
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
            num_inference_steps=self.args.num_inference_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
        )

        # Print policy information
        self.print_policy_info()
        print(
            f"  - obs horizon: {self.args.obs_horizon}, action horizon: {self.args.action_horizon}\n"
            f"  - image size: {self.args.image_size}\n"
            f"  - backbone: {self.args.model_name}, aggregation: {obs_encoder.feature_aggregation}\n"
            f"  - pose repr (state/action): {self.args.state_pose_repr} / {self.args.action_pose_repr}"
        )

        self.policy.set_normalizer(self.fit_normalizer(self.train_dataloader.dataset))
        self.policy.cuda()
        self.policy.train()

        if self.args.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = EMAModel(model=self.ema_policy, power=0.75)

        self.optimizer = torch.optim.AdamW(
            params=self.policy.parameters(),
            lr=self.args.lr,
            betas=[0.95, 0.999],
            eps=1e-8,
            weight_decay=self.args.weight_decay,
        )
        optimizer_to(self.optimizer, "cuda")

        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=2000,
            num_training_steps=len(self.train_dataloader) * self.args.num_epochs,
        )

        self.load_ckpt()

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            batch_result_list = []
            for data in self.train_dataloader:
                loss = self.policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                if self.args.use_ema:
                    self.ema.step(self.policy)
                batch_result_list.append(
                    self.detach_batch_result(
                        {"loss": loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    )
                )
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            policy = self.ema_policy if self.args.use_ema else self.policy
            policy.eval()
            with torch.inference_mode():
                batch_result_list = []
                for data in self.val_dataloader:
                    loss = policy.compute_loss(dict_apply(data, lambda x: x.cuda()))
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                self.update_best_ckpt(epoch_summary, policy=policy)
            policy.train()

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=policy)

        self.save_current_ckpt("last", policy=policy)
        self.save_best_ckpt()

