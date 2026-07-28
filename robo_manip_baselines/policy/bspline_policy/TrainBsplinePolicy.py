import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from diffusion_policy.common.pytorch_util import optimizer_to  # noqa: E402
from diffusion_policy.model.common.lr_scheduler import get_scheduler  # noqa: E402
from diffusion_policy.model.diffusion.ema_model import EMAModel  # noqa: E402

from robo_manip_baselines.common import DataKey, TrainBase  # noqa: E402

from . import BsplineAdapter as adapter  # noqa: E402
from .BsplinePolicyDataset import BsplinePolicyDataset  # noqa: E402
from .BsplineUnetPolicy import BsplineUnetPolicy  # noqa: E402


class TrainBsplinePolicy(TrainBase):
    """Train B-spline policy (BSP) on RMB-format demonstrations.

    Wraps RMB's Diffusion Policy backbone. The only structural difference from
    ``TrainDiffusionPolicy`` is what the network regresses: a
    ``(chunk_size + 2*degree, 1 + action_dim)`` B-spline parameter matrix rather
    than a chunk of discrete actions.
    """

    DatasetClass = BsplinePolicyDataset

    def setup_args(self):
        super().setup_args()

        # horizon is a derived quantity for BSP, not a free hyperparameter:
        # a segment is chunk_size knots plus degree boundary-support knots at
        # each end.
        self.args.horizon = self.args.chunk_size + 2 * self.args.bspline_degree

        if self.args.gripper_action_idxes is None:
            self.args.gripper_action_idxes = list(
                adapter.DEFAULT_GRIPPER_ACTION_IDXES
            )

        # Default the cache alongside the dataset rather than the checkpoint
        # dir, so a fit is reused across training runs. "" disables caching.
        if self.args.bspline_cache_dir is None:
            self.bspline_cache_dir = os.path.join(
                self.args.dataset_dir, ".bspline_cache"
            )
        elif self.args.bspline_cache_dir == "":
            self.bspline_cache_dir = None
        else:
            self.bspline_cache_dir = self.args.bspline_cache_dir

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)
        parser.set_defaults(norm_type="limits")
        parser.set_defaults(batch_size=64)
        parser.set_defaults(num_epochs=500)
        parser.set_defaults(lr=1e-4)
        parser.set_defaults(camera_names=["hand"])
        # The fit runs at full rate regardless; skip only strides observations.
        parser.set_defaults(skip=3)

        parser.add_argument(
            "--weight_decay", type=float, default=1e-6, help="weight decay"
        )
        parser.add_argument(
            "--use_ema",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="enable or disable exponential moving average (EMA)",
        )
        parser.add_argument(
            "--n_obs_steps", type=int, default=2, help="number of observation steps"
        )
        parser.add_argument(
            "--image_size",
            type=int,
            nargs=2,
            default=[320, 240],
            help="image size (width, height)",
        )
        parser.add_argument(
            "--image_crop_size",
            type=int,
            nargs=2,
            default=[288, 216],
            help="image crop size (width, height)",
        )

        # ---- B-spline specific ----
        parser.add_argument(
            "--chunk_size",
            type=int,
            default=10,
            help="number of interior knots per predicted segment "
            "(horizon = chunk_size + 2 * bspline_degree)",
        )
        parser.add_argument(
            "--bspline_degree", type=int, default=3, help="B-spline degree (cubic)"
        )
        parser.add_argument(
            "--max_error",
            type=float,
            default=0.002,
            help="fitting tolerance in weighted action units; for joints this is "
            "radians (0.002 rad = 0.115 deg)",
        )
        parser.add_argument(
            "--bspline_stride",
            type=int,
            default=1,
            help="knot-index step between consecutive segments",
        )
        parser.add_argument(
            "--relative_knots",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="encode knots as first-valid-knot plus differences",
        )
        parser.add_argument(
            "--gripper_action_idxes",
            type=int,
            nargs="*",
            default=None,
            help="indexes of gripper channels within the action vector "
            "(default: [6], the UR5e single-arm layout)",
        )
        parser.add_argument(
            "--bspline_cache_dir",
            type=str,
            default=None,
            help="where to cache fitted B-spline chunks (default: a "
            ".bspline_cache dir alongside the dataset, so the fit is reused "
            "across training runs). The cache key hashes the file list and "
            "every fit parameter, so it invalidates itself; pass an empty "
            "string to disable caching",
        )
        parser.add_argument(
            "--gripper_weight",
            type=float,
            default=adapter.DEFAULT_GRIPPER_WEIGHT,
            help="per-channel fit weight for gripper channels. Effective "
            "tolerance is max_error / gripper_weight, so the default 1e-3 with "
            "max_error 0.002 gives a 2-count tolerance (~0.4 mm on a Hand-e). "
            "Needed because RMB packs radians and Robotiq counts into one vector",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["image_size"] = self.args.image_size
        self.model_meta_info["data"]["image_crop_size"] = self.args.image_crop_size
        self.model_meta_info["data"]["horizon"] = self.args.horizon
        self.model_meta_info["data"]["n_obs_steps"] = self.args.n_obs_steps
        self.model_meta_info["data"]["n_action_steps"] = self.args.horizon

        self.model_meta_info["data"]["bspline"] = {
            "chunk_size": self.args.chunk_size,
            "degree": self.args.bspline_degree,
            "max_error": self.args.max_error,
            "stride": self.args.bspline_stride,
            "relative_knots": self.args.relative_knots,
            "gripper_action_idxes": list(self.args.gripper_action_idxes),
            "gripper_weight": self.args.gripper_weight,
            # filled in by set_data_stats, once the action dimension is known
            "action_weights": None,
            "action_dim": None,
            "origin_time_scale": None,
        }

        self.model_meta_info["policy"]["use_ema"] = self.args.use_ema
        self.model_meta_info["policy"]["backbone"] = "cnn"
        self.model_meta_info["policy"]["scheduler"] = "ddim"

    def get_extra_norm_config(self):
        if self.args.norm_type == "limits":
            return {"out_min": -1.0, "out_max": 1.0}
        return super().get_extra_norm_config()

    def set_data_stats(self):
        """State/image stats as usual; action stats from the fitted splines.

        ``TrainBase.set_data_stats`` computes per-element stats over the raw
        action sequence. For BSP the action is a parameter matrix, and its stats
        must be **per-channel replicated across rows** or normalization destroys
        the knot column's monotonicity (see BsplineAdapter.build_action_stats).
        So the base implementation runs for state and images, then the action
        block is replaced wholesale.
        """
        super().set_data_stats()

        bspline_info = self.model_meta_info["data"]["bspline"]

        # Time base: measured, never assumed. The knots the policy predicts are
        # in frames, and this is the frames-per-second factor the rollout needs
        # to convert them to wall-clock.
        time_info = adapter.measure_time_base(self.all_filenames)
        bspline_info["origin_time_scale"] = time_info["origin_time_scale"]
        bspline_info["time_base_info"] = time_info

        action_dim = len(self.model_meta_info["action"]["example"])
        bspline_info["action_dim"] = action_dim

        if action_dim != 7 and self.args.gripper_action_idxes == list(
            adapter.DEFAULT_GRIPPER_ACTION_IDXES
        ):
            raise ValueError(
                f"[{self.__class__.__name__}] action_dim is {action_dim}, but "
                f"--gripper_action_idxes was left at the UR5e default "
                f"{list(adapter.DEFAULT_GRIPPER_ACTION_IDXES)}. Pass the gripper "
                f"channel indexes explicitly for this robot."
            )

        weights = adapter.build_action_weights(
            action_dim, self.args.gripper_action_idxes, self.args.gripper_weight
        )
        bspline_info["action_weights"] = weights.tolist()

        # Recorded so the dataset (which refits per train/val split) reuses the
        # same cache rather than repeating the fit.
        bspline_info["cache_dir"] = self.bspline_cache_dir
        fitter = adapter.build_fitter(
            self.all_filenames,
            self.model_meta_info["action"]["keys"],
            bspline_info,
            cache_dir=self.bspline_cache_dir,
        )

        bspline_info["compression_ratio"] = float(fitter.compression_ratio)
        self.model_meta_info["action"] = {
            **self.model_meta_info["action"],
            **adapter.build_action_stats(
                fitter,
                self.args.norm_type,
                **self.get_extra_norm_config(),
            ),
        }

        print(
            f"[{self.__class__.__name__}] B-spline fit: "
            f"compression {fitter.compression_ratio:.2f}x, "
            f"target shape {fitter.all_actions.shape[1:]}, "
            f"origin_time_scale {bspline_info['origin_time_scale']:.2f} Hz"
        )

    def setup_policy(self):
        state_dim = len(self.model_meta_info["state"]["example"])
        action_dim = self.model_meta_info["data"]["bspline"]["action_dim"]

        if len(self.args.camera_names) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] camera_names must be non-empty; "
                f"the B-spline policy wraps the image-conditioned backbone."
            )

        shape_meta = {
            "obs": {},
            # The physical action dim. BsplineUnetPolicy widens it by one for
            # the knot column.
            "action": {"shape": [action_dim]},
        }
        if len(self.args.state_keys) > 0:
            shape_meta["obs"]["state"] = {"shape": [state_dim], "type": "low_dim"}
        for camera_name in self.args.camera_names:
            shape_meta["obs"][DataKey.get_rgb_image_key(camera_name)] = {
                "shape": [3, self.args.image_size[1], self.args.image_size[0]],
                "type": "rgb",
            }

        self.model_meta_info["policy"]["args"] = {
            "shape_meta": shape_meta,
            "horizon": self.args.horizon,
            # The whole segment is returned; BSP has no notion of "execute the
            # first k actions of the chunk".
            "n_action_steps": self.args.horizon,
            "n_obs_steps": self.args.n_obs_steps,
            "crop_shape": self.args.image_crop_size[::-1],  # (height, width)
            "obs_encoder_group_norm": True,
            "eval_fixed_crop": True,
            "num_inference_steps": 8,
            "down_dims": [256, 512, 1024],
            "obs_as_global_cond": True,
            "diffusion_step_embed_dim": 128,
            "kernel_size": 5,
            "n_groups": 8,
            "cond_predict_scale": True,
            "bspline_degree": self.args.bspline_degree,
        }
        self.model_meta_info["policy"]["noise_scheduler_args"] = {
            "beta_end": 0.02,
            "beta_schedule": "squaredcos_cap_v2",
            "beta_start": 0.0001,
            # Sound only because targets are normalized to [-1, 1]; see
            # BsplineAdapter.build_action_stats.
            "clip_sample": True,
            "num_train_timesteps": 100,
            "prediction_type": "epsilon",
            "set_alpha_to_one": True,
            "steps_offset": 0,
        }

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        noise_scheduler = DDIMScheduler(
            **self.model_meta_info["policy"]["noise_scheduler_args"]
        )

        self.policy = BsplineUnetPolicy(
            noise_scheduler=noise_scheduler,
            **self.model_meta_info["policy"]["args"],
        )

        if self.args.use_ema:
            self.ema_policy = copy.deepcopy(self.policy)
            self.ema = EMAModel(
                model=self.ema_policy,
                update_after_step=0,
                inv_gamma=1.0,
                power=0.75,
                min_value=0.0,
                max_value=0.9999,
            )

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.95, 0.999),
            eps=1e-8,
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        self.policy.cuda()
        if self.args.use_ema:
            self.ema_policy.cuda()
        optimizer_to(self.optimizer, "cuda")

        self.print_policy_info()
        bspline_info = self.model_meta_info["data"]["bspline"]
        print(f"  - use ema: {self.args.use_ema}, backbone: cnn, scheduler: ddim")
        print(
            f"  - horizon: {self.args.horizon} "
            f"(chunk_size {self.args.chunk_size} + 2*degree {self.args.bspline_degree}), "
            f"obs steps: {self.args.n_obs_steps}"
        )
        print(
            f"  - action channels: {action_dim + 1} "
            f"(1 knot + {action_dim} control point dims)"
        )
        print(
            f"  - max_error: {self.args.max_error}, "
            f"gripper weight: {self.args.gripper_weight} "
            f"(effective {self.args.max_error / self.args.gripper_weight:.2f} counts)"
        )
        print(
            f"  - compression: {bspline_info['compression_ratio']:.2f}x, "
            f"origin_time_scale: {bspline_info['origin_time_scale']:.2f} Hz"
        )
        print(
            f"  - image size: {self.args.image_size}, "
            f"image crop size: {self.args.image_crop_size}"
        )

    def print_policy_info(self):
        # TrainBase derives action dim from len(action["example"]), which for BSP
        # is the number of parameter rows (16), not the action dimension (7).
        bspline_info = self.model_meta_info["data"]["bspline"]
        print(
            f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
            f"  - state dim: {len(self.model_meta_info['state']['example'])}, "
            f"action dim: {bspline_info['action_dim']}, "
            f"target shape: {tuple(self.model_meta_info['action']['example'].shape)}, "
            f"camera num: {len(self.args.camera_names)}\n"
            f"  - state keys: {self.args.state_keys}\n"
            f"  - action keys: {self.args.action_keys}\n"
            f"  - camera names: {self.args.camera_names}\n"
            f"  - skip: {self.args.skip} (observations only; the spline is fitted "
            f"at full rate), batch size: {self.args.batch_size}, "
            f"num epochs: {self.args.num_epochs}, num workers: {self.args.num_workers}"
        )

    def load_ckpt(self):
        super().load_ckpt()

        if self.args.pretrain_checkpoint is not None and self.args.use_ema:
            self.ema_policy.load_state_dict(self.policy.state_dict())

    def train_loop(self):
        from diffusion_policy.common.pytorch_util import dict_apply
        from tqdm import tqdm

        for epoch in tqdm(range(self.args.num_epochs)):
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

            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>4}", policy=policy)

        self.save_current_ckpt("last", policy=policy)
        self.save_best_ckpt()
