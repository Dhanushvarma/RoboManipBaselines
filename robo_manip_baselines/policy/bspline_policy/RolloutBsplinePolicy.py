import os
import queue
import sys
import threading
import time

import cv2
import matplotlib.pylab as plt
import numpy as np
import torch

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../third_party/diffusion_policy")
)
from robo_manip_baselines.common import (  # noqa: E402
    DataKey,
    RolloutBase,
    convert_data_to_policy,
    denormalize_data,
    normalize_data,
)

from .BSplineAction import eval_bspline_at  # noqa: E402
from .BsplineUnetPolicy import BsplineUnetPolicy  # noqa: E402


class RolloutBsplinePolicy(RolloutBase):
    """Algorithm 2: pipelined B-spline rollout.

    RMB's rollout loop is synchronous and single-rate, which Algorithm 2 is not.
    The split is:

    * this class runs in RMB's loop and decides *when* to replan, running
      inference on a worker thread so the loop never blocks on the GPU;
    * the env owns a 100 Hz thread that samples whatever segment is installed
      and feeds ``JointInterpolationController``.

    Replanning fires on plan exhaustion (as in the reference implementation)
    **or** on ``--max_plan_age``, whichever comes first. The age trigger is an
    addition: measured on real UR5e data, fitted segments span ~1.2 s, so
    exhaustion alone would leave the policy open-loop for over a second. Segment
    alignment already makes installing a segment mid-plan safe, so this costs
    only inference.
    """

    def set_additional_args(self, parser):
        parser.add_argument(
            "--speedup",
            type=float,
            default=1.0,
            help="temporal rescaling factor; the segment is traversed this much "
            "faster than the demonstration (a_exec(t) = a(m*t))",
        )
        parser.add_argument(
            "--max_plan_age",
            type=float,
            default=0.4,
            help="force a replan when the live segment is older than this [s], "
            "even if it has not been exhausted. Set <= 0 to disable and replan "
            "on exhaustion only, as in the reference implementation",
        )
        parser.add_argument(
            "--predict_before_end",
            type=float,
            default=0.1,
            help="replan when this much wall-clock time of the segment remains",
        )
        parser.add_argument(
            "--no_segment_align",
            action="store_true",
            help="disable inference-time segment alignment (Eq. 2)",
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="run with no hardware; print the joint targets and timing that "
            "would be sent through callm_controller",
        )
        parser.add_argument(
            "--num_inference_steps",
            type=int,
            default=None,
            help="override the checkpoint's DDIM step count (the reference "
            "deploys at 10 against 16 at train time)",
        )

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        if "bspline" not in self.model_meta_info["data"]:
            raise ValueError(
                f"[{self.__class__.__name__}] Checkpoint has no B-spline metadata; "
                f"was it trained with BsplinePolicy?"
            )
        self.bspline_info = self.model_meta_info["data"]["bspline"]

        # RolloutBase derives action_dim from len(action["example"]), which for
        # BSP is the parameter-matrix row count, not the action dimension.
        self.action_dim = int(self.bspline_info["action_dim"])
        self.action_weights = np.asarray(
            self.bspline_info["action_weights"], dtype=np.float64
        )
        self.bspline_degree = int(self.bspline_info["degree"])
        self.relative_knots = bool(self.bspline_info["relative_knots"])
        self.origin_time_scale = float(self.bspline_info["origin_time_scale"])

        if self.args.max_plan_age is not None and self.args.max_plan_age > 0:
            # A segment spanning S seconds at 1x is consumed in S/m at speedup m,
            # so past a certain speedup exhaustion always wins and the age
            # trigger is inert. Not an error, but worth saying out loud.
            print(
                f"[{self.__class__.__name__}] max_plan_age "
                f"{self.args.max_plan_age:.2f}s at speedup {self.args.speedup:.1f}x "
                f"covers {self.args.max_plan_age * self.args.speedup:.2f}s of plan time."
            )

    def setup_policy(self):
        policy_args = dict(self.model_meta_info["policy"]["args"])
        if self.args.num_inference_steps is not None:
            policy_args["num_inference_steps"] = self.args.num_inference_steps

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        noise_scheduler = DDIMScheduler(
            **self.model_meta_info["policy"]["noise_scheduler_args"]
        )
        self.policy = BsplineUnetPolicy(
            noise_scheduler=noise_scheduler, **policy_args
        )
        self.load_ckpt()

        self.print_policy_info()
        print(
            f"  - origin_time_scale: {self.origin_time_scale:.2f} Hz, "
            f"speedup: {self.args.speedup:.2f}x, "
            f"max_plan_age: {self.args.max_plan_age:.2f}s, "
            f"align: {not self.args.no_segment_align}"
        )
        print(
            f"  - inference steps: {policy_args['num_inference_steps']}, "
            f"target shape: ({policy_args['horizon']}, {self.action_dim + 1})"
        )

    def setup_plot(self, fig_ax=None):
        if fig_ax is None:
            fig_ax = plt.subplots(
                2, max(len(self.camera_names), 1), figsize=(13.5, 6.0), dpi=60, squeeze=False
            )
        super().setup_plot(fig_ax)

    def setup_variables(self):
        super().setup_variables()
        self._inference_queue = queue.Queue(maxsize=1)
        self._inference_thread = None
        self._in_flight = False
        self._in_flight_lock = threading.Lock()
        self._epoch = 0

    def reset_variables(self):
        super().reset_variables()
        self.state_buf = None
        self.images_buf = None
        self._epoch += 1
        self.inference_count = 0
        self.align_errors = []
        self.inference_latencies = []
        if hasattr(self.env.unwrapped, "stop_bspline"):
            self.env.unwrapped.stop_bspline()

    # ------------------------------------------------------------ inference --

    def infer_policy(self):
        if len(self.state_keys) > 0:
            self.update_state_buf()
        if len(self.camera_names) > 0:
            self.update_images_buf()

        self._start_inference_thread()

        if self._needs_replan():
            self._submit_inference()

        self._drain_completed_inference()

        # Keep RMB's bookkeeping (recording, plotting, set_command_data) fed
        # with the joint vector actually being commanded.
        last = getattr(self.env.unwrapped, "_last_command", None)
        if last is None:
            last = self.motion_manager.get_data(DataKey.COMMAND_JOINT_POS, self.obs)
        self.policy_action = np.asarray(last, dtype=np.float64)
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def _needs_replan(self):
        with self._in_flight_lock:
            if self._in_flight:
                return False

        state = self.env.unwrapped.get_segment_state()
        if state is None:
            return True

        if state["seconds_remaining"] < self.args.predict_before_end * self.args.speedup:
            return True
        if self.args.max_plan_age > 0 and state["age"] > self.args.max_plan_age:
            return True
        return False

    def _submit_inference(self):
        if len(self.camera_names) > 0:
            input_data = {}
            if len(self.state_keys) > 0:
                input_data["state"] = self.get_state()
            for camera_name, image in zip(self.camera_names, self.get_images()):
                input_data[DataKey.get_rgb_image_key(camera_name)] = image
        else:
            input_data = {"obs": self.get_state()}

        request = {
            "input": input_data,
            "obs_time": time.perf_counter(),
            "epoch": self._epoch,
        }
        try:
            self._inference_queue.put_nowait(request)
            with self._in_flight_lock:
                self._in_flight = True
        except queue.Full:
            pass

    def _start_inference_thread(self):
        if self._inference_thread is not None and self._inference_thread.is_alive():
            return
        self._result_queue = queue.Queue()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, name="BSplineInference", daemon=True
        )
        self._inference_thread.start()

    def _inference_loop(self):
        while True:
            request = self._inference_queue.get()
            try:
                with torch.inference_mode():
                    params = self.policy.predict_action(request["input"])["action"][0]
                params = params.cpu().detach().numpy().astype(np.float64)
                self._result_queue.put(
                    {
                        "params": params,
                        "latency": time.perf_counter() - request["obs_time"],
                        "epoch": request["epoch"],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - surface, never wedge the loop
                print(f"[{self.__class__.__name__}] Inference failed: {exc}")
                self._result_queue.put(None)

    def _drain_completed_inference(self):
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            return

        with self._in_flight_lock:
            self._in_flight = False

        if result is None or result["epoch"] != self._epoch:
            return

        params = denormalize_data(result["params"], self.model_meta_info["action"])
        installed = self.env.unwrapped.install_bspline_segment(
            params,
            origin_time_scale=self.origin_time_scale,
            speedup=self.args.speedup,
            degree=self.bspline_degree,
            weights=self.action_weights,
            relative_knots=self.relative_knots,
            inference_latency=result["latency"],
            align=not self.args.no_segment_align,
        )
        if installed:
            self.inference_count += 1
            self.inference_latencies.append(result["latency"])
            state = self.env.unwrapped.get_segment_state()
            if state is not None and state["align_error"] is not None:
                self.align_errors.append(state["align_error"])

    # ---------------------------------------------------------- observation --

    def update_state_buf(self):
        state = np.concatenate(
            [
                convert_data_to_policy(
                    self.motion_manager.get_data(state_key, self.obs), state_key
                )
                for state_key in self.state_keys
            ]
        )
        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state, dtype=torch.float32)

        if self.state_buf is None:
            self.state_buf = [
                state for _ in range(self.model_meta_info["data"]["n_obs_steps"])
            ]
        else:
            self.state_buf.pop(0)
            self.state_buf.append(state)

    def get_state(self):
        return torch.stack(self.state_buf, dim=0)[torch.newaxis].to(self.device)

    def update_images_buf(self):
        images = []
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name]
            image = cv2.resize(image, tuple(self.model_meta_info["data"]["image_size"]))
            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = self.image_transforms(image)
            image = image * 2.0 - 1.0
            images.append(image)

        if self.images_buf is None:
            self.images_buf = [
                [image for _ in range(self.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(self.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

    def get_images(self):
        return [
            torch.stack(single_images_buf, dim=0)[torch.newaxis].to(self.device)
            for single_images_buf in self.images_buf
        ]

    # --------------------------------------------------------------- output --

    def draw_plot(self):
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        if len(self.camera_names) > 0:
            self.plot_images(self.ax[0, 0 : len(self.camera_names)])
            self.plot_action(self.ax[1, 0])
        else:
            self.plot_action(self.ax[0, 0])

        self.fig.tight_layout()
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    def print_statistics(self):
        super().print_statistics()

        if getattr(self, "inference_count", 0):
            print(f"[{self.__class__.__name__}] B-spline rollout statistics:")
            print(f"  - segments installed: {self.inference_count}")
            if self.inference_latencies:
                lat = np.array(self.inference_latencies)
                # RolloutBase's own "Inference duration" now measures only the
                # async submit, so report the real end-to-end latency here.
                print(
                    f"  - inference latency [s]: mean {lat.mean():.4f} "
                    f"p95 {np.percentile(lat, 95):.4f} max {lat.max():.4f}"
                )
                print(
                    f"  - effective replan rate: "
                    f"{self.inference_count / max(sum(self.result['duration']), 1e-6):.2f} Hz"
                )
            if self.align_errors:
                errors = np.array(self.align_errors)
                print(
                    f"  - segment alignment error [rad]: "
                    f"mean {errors.mean():.4f} p95 {np.percentile(errors, 95):.4f} "
                    f"max {errors.max():.4f}"
                )
            print(f"  - sampler loop: {self.env.unwrapped.sampler_stats}")

        if getattr(self.args, "dry_run", False):
            self.env.unwrapped.print_dry_run_summary()
