"""Real UR5e environment for B-spline Policy, driven by ``callm_controller``.

Differs from :class:`RealUR5eEnvBase` in one structural way: the arm is not
commanded once per ``env.step()``. A B-spline segment is installed by the policy
at roughly 1-3 Hz and a daemon thread samples it at ``spline_rate`` (100 Hz by
default), feeding ``JointInterpolationController``'s servoJ loop. That is
Algorithm 2 of the paper -- the spline *is* the interpolator, evaluated against
the wall clock, with no waypoint queue and no blending.

``ur_rtde`` is never imported here; the arm is reached through
``callm_controller``'s ``JointInterpolationController`` (and that import is
deferred so ``--dry_run`` needs no robot drivers at all). The ``CallM``
composite is deliberately not used: it also owns a TriOrb mobile base, which is
out of scope and would add a ``triorb-core`` dependency.
"""

import threading
import time
from os import path

import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig

from ..RealEnvBase import RealEnvBase


class LoopStats:
    """Per-phase timing with an overrun counter.

    Ported from the reference implementation's ``mock_env.py``. The point is to
    show, before touching hardware, that the sampler actually keeps up: an
    overrun means a tick took longer than its period, so commands were late.
    """

    def __init__(self, period):
        self.period = float(period)
        self.samples = []
        self.overruns = 0

    def add(self, duration):
        self.samples.append(duration)
        if duration > self.period:
            self.overruns += 1

    def summary(self):
        if not self.samples:
            return {"n": 0}
        arr = np.array(self.samples)
        return {
            "n": len(arr),
            "mean_ms": float(arr.mean() * 1e3),
            "p50_ms": float(np.percentile(arr, 50) * 1e3),
            "p95_ms": float(np.percentile(arr, 95) * 1e3),
            "max_ms": float(arr.max() * 1e3),
            "overruns": self.overruns,
            "overrun_pct": 100.0 * self.overruns / len(arr),
        }

    def __str__(self):
        s = self.summary()
        if s["n"] == 0:
            return "no samples"
        return (
            f"n={s['n']} mean={s['mean_ms']:.2f}ms p50={s['p50_ms']:.2f}ms "
            f"p95={s['p95_ms']:.2f}ms max={s['max_ms']:.2f}ms "
            f"overruns={s['overruns']} ({s['overrun_pct']:.1f}%)"
        )


class DryRunArm:
    """Hardware-free stand-in for ``JointInterpolationController``.

    Mirrors the method surface used here and logs every command that would have
    been sent, so a rollout can be inspected end to end with no robot attached.
    Imports nothing from ``ur_rtde``.
    """

    def __init__(self, joints_init=None, verbose=True, log_limit=None):
        self._q = (
            np.array(joints_init, dtype=np.float64)
            if joints_init is not None
            else np.zeros(6)
        )
        self.verbose = verbose
        self.log_limit = log_limit
        self.commands = []

    def start(self, wait=True):
        print("[DryRunArm] started (no hardware)")

    def stop(self, wait=True):
        print(f"[DryRunArm] stopped after {len(self.commands)} arm commands")

    def servoJ(self, joints, duration=0.1):
        self._q = np.array(joints, dtype=np.float64)
        record = (time.time(), self._q.copy(), float(duration))
        if self.log_limit is None or len(self.commands) < self.log_limit:
            self.commands.append(record)

    def schedule_waypoint(self, joints, target_time):
        self.servoJ(joints, duration=max(target_time - time.time(), 1e-3))

    def get_state(self, k=None):
        return {
            "ActualQ": self._q.copy(),
            "ActualQd": np.zeros(6),
            "ActualTCPPose": np.zeros(6),
            "ActualTCPForce": np.zeros(6),
            "TargetQ": self._q.copy(),
            "robot_receive_timestamp": time.time(),
        }

    def zero_ft_sensor(self):
        pass


class DryRunGripper:
    """Hardware-free Robotiq stand-in."""

    def __init__(self):
        self._pos = 0.0
        self.commands = []

    def connect(self, *args, **kwargs):
        pass

    def disconnect(self):
        pass

    def activate(self):
        pass

    def move(self, position, speed, force):
        self._pos = float(position)
        self.commands.append((time.time(), self._pos))

    def get_current_position(self):
        return self._pos


class DryRunCamera:
    """Synthetic RGB-D source so ``--dry_run`` needs no RealSense attached.

    Matches the slice of the RealSense interface ``RealEnvBase.get_camera_data``
    uses. The frame counter is baked into the blue channel, mirroring the
    reference implementation's ``mock_env.py``, so it is visually obvious
    whether frames are advancing.
    """

    def __init__(self):
        self.color_fovy = 55.0
        self.depth_fovy = 55.0
        self._frame = 0

    def read(self, image_size):
        width, height = image_size
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        rgb[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        rgb[:, :, 2] = self._frame % 256
        depth = np.full((height, width, 3), 1000, dtype=np.uint16)
        self._frame += 1
        return rgb, depth


class RealUR5eBSplineEnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [-2 * np.pi, -2 * np.pi, -1 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi, 0.0],
            dtype=np.float32,
        ),
        high=np.array(
            [2 * np.pi, 2 * np.pi, 1 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi, 255.0],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip,
        camera_ids,
        pointcloud_camera_ids,
        gelsight_ids,
        sanwa_keyboard_ids,
        init_qpos,
        # ---- callm_controller arm ----
        arm_frequency=125,
        # Low end of the permitted [0.03, 0.2]. CallM's default of 0.2 would
        # visibly lag a 100 Hz command stream: the B-spline, the joint
        # interpolator and servoJ's own lookahead all smooth, and we only want
        # the last of those to be gentle.
        arm_lookahead_time=0.03,
        arm_gain=100,
        max_joint_speed=1.05,
        # ---- B-spline sampling ----
        spline_rate=100.0,
        gripper_rate=20.0,
        # ---- safety ----
        stale_plan_timeout=1.0,
        # ---- gripper ----
        gripper_port=63352,
        gripper_speed=50,
        gripper_force=10,
        dry_run=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.init_qpos = np.array(init_qpos, dtype=np.float64)
        self.joint_vel_limit = np.deg2rad(191)  # [rad/s]
        self.dry_run = bool(dry_run)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf"
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

        self.robot_ip = robot_ip
        self.arm_frequency = int(arm_frequency)
        self.max_joint_speed = float(max_joint_speed)
        self.spline_rate = float(spline_rate)
        self.spline_period = 1.0 / self.spline_rate
        self.gripper_period = 1.0 / float(gripper_rate)
        self.stale_plan_timeout = float(stale_plan_timeout)
        self.gripper_speed = int(gripper_speed)
        self.gripper_force = int(gripper_force)

        # Safety envelope. Driving the arm from the sampler thread bypasses
        # RealEnvBase.overwrite_command_for_safety, so the equivalent guards are
        # reimplemented in _clamp_arm_command below.
        self.joint_pos_low = self.action_space.low[:6].astype(np.float64)
        self.joint_pos_high = self.action_space.high[:6].astype(np.float64)
        self.max_joint_step = self.max_joint_speed * self.spline_period

        # ---- connect ----
        if self.dry_run:
            print(f"[{self.__class__.__name__}] DRY RUN: no hardware will be touched.")
            self.arm = DryRunArm(joints_init=self.init_qpos[:6])
            self.gripper = DryRunGripper()
            self._shm_manager = None
        else:
            from multiprocessing.managers import SharedMemoryManager

            from callm_controller.robot.joint_interpolation_controller import (
                JointInterpolationController,
            )
            from callm_controller.robot.robotiq_gripper import RobotiqGripper

            print(f"[{self.__class__.__name__}] Start connecting the UR5e.")
            self._shm_manager = SharedMemoryManager()
            self._shm_manager.start()
            self.arm = JointInterpolationController(
                shm_manager=self._shm_manager,
                robot_ip=robot_ip,
                frequency=self.arm_frequency,
                lookahead_time=arm_lookahead_time,
                gain=arm_gain,
                max_joint_speed=self.max_joint_speed,
            )
            self.arm.start(wait=True)
            print(f"[{self.__class__.__name__}] Finish connecting the UR5e.")

            print(f"[{self.__class__.__name__}] Start connecting the Robotiq gripper.")
            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=int(gripper_port))
            print(f"[{self.__class__.__name__}] Finish connecting the Robotiq gripper.")

        self._gripper_activated = False
        self.arm_joint_pos_actual = np.array(self.arm.get_state()["ActualQ"])

        # ---- spline segment state (shared with the sampler thread) ----
        self._segment_lock = threading.Lock()
        self._segment = None  # dict, see install_bspline_segment
        self._last_command = None  # last commanded 7-vector
        self._stop_event = threading.Event()
        self._sampler_thread = None
        self._last_gripper_time = 0.0
        self._last_gripper_sent = 0.0
        self.sampler_stats = LoopStats(self.spline_period)
        self.plan_exhausted = False

        if self.dry_run:
            # Synthesize whatever cameras the config asked for, so the same
            # config drives a dry run and a real rollout.
            for camera_name in (camera_ids or {}):
                self.cameras[camera_name] = DryRunCamera()
            if self.cameras:
                print(
                    f"[{self.__class__.__name__}] DRY RUN cameras: "
                    f"{list(self.cameras.keys())}"
                )
        else:
            self.setup_realsense(camera_ids)
            self.setup_femtobolt(pointcloud_camera_ids)
            self.setup_gelsight(gelsight_ids)
            self.setup_sanwa_keyboard(sanwa_keyboard_ids)

    # --------------------------------------------------------- BSP interface --

    def install_bspline_segment(
        self,
        params,
        origin_time_scale,
        speedup=1.0,
        degree=3,
        weights=None,
        relative_knots=False,
        inference_latency=0.0,
        align=True,
    ):
        """Install a freshly predicted segment, aligning it to what is executing.

        ``params`` is the denormalized ``(chunk_size + 2*degree, 1 + action_dim)``
        matrix. Alignment (Eq. 2 of the paper) picks the parameter value on the
        new curve whose joint vector best matches the last command, so the seam
        is continuous in value even though the plans are independent.
        """
        from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
            bspline_span,
            eval_bspline_at,
            project_knots_monotonic,
        )

        params = project_knots_monotonic(params, degree=degree)
        t_min, t_max = bspline_span(params, degree=degree, relative_knots=relative_knots)
        if t_max <= t_min:
            print(
                f"[{self.__class__.__name__}] Rejecting degenerate segment "
                f"(span [{t_min}, {t_max}])."
            )
            return False

        weights = (
            np.ones(params.shape[1] - 1) if weights is None else np.asarray(weights)
        )

        u_start = t_min
        align_error = None
        if align and self._last_command is not None:
            u_start, align_error = self._align_segment(
                params,
                t_min,
                t_max,
                degree,
                weights,
                relative_knots,
                inference_latency,
                origin_time_scale,
                speedup,
            )

        with self._segment_lock:
            self._segment = {
                "params": params,
                "degree": degree,
                "weights": weights,
                "relative_knots": relative_knots,
                "t_min": t_min,
                "t_max": t_max,
                "origin_time_scale": float(origin_time_scale),
                "speedup": float(speedup),
                # Back-date the origin so that right now u == u_start.
                "plan_origin": time.perf_counter()
                - u_start / (float(speedup) * float(origin_time_scale)),
                "installed_at": time.perf_counter(),
                "align_error": align_error,
            }
            self.plan_exhausted = False

        self._ensure_sampler_running()
        return True

    def _align_segment(
        self,
        params,
        t_min,
        t_max,
        degree,
        weights,
        relative_knots,
        inference_latency,
        origin_time_scale,
        speedup,
    ):
        """Eq. 2: t* = argmin MSE(S_new(t), a_last) over a bounded window.

        Follows the reference implementation rather than the paper: the window
        upper bound is the *measured* inference latency converted to frames,
        grown geometrically until the match is good enough, and hard-capped at
        20% into the plan so alignment can never skip most of a fresh segment.
        The gripper channel is excluded from the distance, matching their
        default (``consider_gripper_during_align=False``).
        """
        from scipy.optimize import minimize_scalar

        from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
            eval_bspline_at,
        )

        target = np.asarray(self._last_command, dtype=np.float64)[:6]
        # Cap: never enter more than 20% into the segment.
        max_allowed = t_min + 0.2 * (t_max - t_min)
        window = float(
            np.clip(inference_latency * speedup * origin_time_scale, t_min, max_allowed)
        )

        def distance(u):
            value = eval_bspline_at(
                params, float(u), degree=degree, relative_knots=relative_knots
            )
            return float(np.abs(value[:6] / weights[:6] - target).max())

        best_u, best_err = t_min, distance(t_min)
        lam = 1.0
        while best_err > 0.1 and lam <= 20.0:
            upper = min(t_min + max(window - t_min, 0.0) * lam, max_allowed)
            if upper <= t_min:
                break
            result = minimize_scalar(
                distance, bounds=(t_min, upper), method="bounded"
            )
            if result.fun < best_err:
                best_u, best_err = float(result.x), float(result.fun)
            lam *= 1.5

        return best_u, best_err

    def stop_bspline(self):
        with self._segment_lock:
            self._segment = None

    def get_segment_state(self):
        """``(u, t_max, seconds_remaining)`` for the live plan, else ``None``."""
        with self._segment_lock:
            seg = self._segment
            if seg is None:
                return None
            u = self._segment_u(seg)
            remaining = (seg["t_max"] - u) / (seg["speedup"] * seg["origin_time_scale"])
            return {
                "u": u,
                "t_max": seg["t_max"],
                "seconds_remaining": remaining,
                "age": time.perf_counter() - seg["installed_at"],
                "align_error": seg["align_error"],
            }

    @staticmethod
    def _segment_u(seg):
        elapsed = time.perf_counter() - seg["plan_origin"]
        return elapsed * seg["speedup"] * seg["origin_time_scale"]

    # ------------------------------------------------------- sampler thread --

    def _ensure_sampler_running(self):
        if self._sampler_thread is not None and self._sampler_thread.is_alive():
            return
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(
            target=self._sampler_loop, name="BSplineSampler", daemon=True
        )
        self._sampler_thread.start()

    def _sampler_loop(self):
        from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
            eval_bspline_at,
        )

        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            tick_start = time.perf_counter()

            with self._segment_lock:
                seg = self._segment

            if seg is not None:
                # Stale-plan watchdog: if nothing new has arrived well after this
                # plan should have finished, hold position rather than sitting on
                # a dead plan's final pose forever.
                #
                # Measured from when the plan *runs out*, not when it was
                # installed: plan length is data-dependent (a policy chunk spans
                # ~1.2 s, a whole-episode segment from the replay diagnostic
                # spans ~18 s), so a fixed timeout from install would kill any
                # long but perfectly healthy plan mid-motion.
                plan_duration = (seg["t_max"] - seg["t_min"]) / (
                    seg["speedup"] * seg["origin_time_scale"]
                )
                age = tick_start - seg["installed_at"]
                if age > plan_duration + self.stale_plan_timeout:
                    print(
                        f"[{self.__class__.__name__}] Stale plan ({age:.2f}s since "
                        f"install, plan spans {plan_duration:.2f}s); holding position."
                    )
                    with self._segment_lock:
                        self._segment = None
                else:
                    u = float(np.clip(self._segment_u(seg), seg["t_min"], seg["t_max"]))
                    if u >= seg["t_max"] - 1e-9:
                        self.plan_exhausted = True
                    value = eval_bspline_at(
                        seg["params"],
                        u,
                        degree=seg["degree"],
                        relative_knots=seg["relative_knots"],
                    )
                    command = np.asarray(value, dtype=np.float64) / seg["weights"]
                    self._send_command(command, tick_start)

            self.sampler_stats.add(time.perf_counter() - tick_start)

            next_tick += self.spline_period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind; resynchronise rather than accumulate debt.
                next_tick = time.perf_counter()

    def _clamp_arm_command(self, arm_command):
        """Joint-limit box plus a per-tick delta clamp.

        These replace ``RealEnvBase.overwrite_command_for_safety``, which the
        sampler path bypasses. ``JointInterpolationController`` also enforces
        ``max_joint_speed`` internally; this is the belt to that pair of braces,
        and it is what stops a bad prediction reaching the interpolator at all.
        """
        if not np.all(np.isfinite(arm_command)):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Arm command not finite: {arm_command}"
            )
        arm_command = np.clip(arm_command, self.joint_pos_low, self.joint_pos_high)
        if self._last_command is not None:
            previous = np.asarray(self._last_command[:6], dtype=np.float64)
            arm_command = previous + np.clip(
                arm_command - previous, -self.max_joint_step, self.max_joint_step
            )
        return arm_command

    def _send_command(self, command, now):
        arm_command = self._clamp_arm_command(command[:6])
        self.arm.servoJ(arm_command, duration=self.spline_period)

        gripper_command = float(np.clip(command[6], 0.0, 255.0))
        if now - self._last_gripper_time >= self.gripper_period:
            self._last_gripper_sent = float(int(gripper_command))
            self.gripper.move(
                int(gripper_command), self.gripper_speed, self.gripper_force
            )
            self._last_gripper_time = now

        # Record the value actually sent, not the value just sampled: the gripper
        # is commanded at gripper_rate (~20 Hz), well below the sampler, so the
        # two differ. Logging the sampled value would make the replay diagnostic
        # report a gripper tracking error that is really just this rate gap.
        self._last_command = np.concatenate([arm_command, [self._last_gripper_sent]])

    # ------------------------------------------------------------- gym API ---

    def _reset_robot(self):
        print(f"[{self.__class__.__name__}] Start moving the robot to the reset position.")
        self.stop_bspline()
        self._last_command = None
        self._set_action(self.init_qpos, duration=None, joint_vel_limit_scale=0.3, wait=True)
        print(f"[{self.__class__.__name__}] Finish moving the robot to the reset position.")

        if not self._gripper_activated:
            self._gripper_activated = True
            print(f"[{self.__class__.__name__}] Start activating the Robotiq gripper.")
            self.gripper.activate()
            print(f"[{self.__class__.__name__}] Finish activating the Robotiq gripper.")

        time.sleep(0.2)
        self.arm.zero_ft_sensor()
        time.sleep(0.2)

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        """Direct command path, used before the policy takes over.

        While a B-spline segment is live the sampler thread owns the arm, so
        this becomes a no-op for it -- otherwise RMB's per-step command would
        fight the 100 Hz stream.
        """
        with self._segment_lock:
            segment_active = self._segment is not None
        if segment_active:
            return

        start_time = time.time()
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        self.arm.servoJ(arm_joint_pos_command, duration=max(duration, 1.0 / self.arm_frequency))

        gripper_pos = float(action[self.body_config_list[0].gripper_joint_idxes][0])
        self.gripper.move(int(gripper_pos), self.gripper_speed, self.gripper_force)
        self._last_gripper_sent = float(int(gripper_pos))
        self._last_command = np.concatenate([arm_joint_pos_command, [gripper_pos]])

        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        state = self.arm.get_state()
        arm_joint_pos = np.array(state["ActualQ"], dtype=np.float64)
        arm_joint_vel = np.array(state["ActualQd"], dtype=np.float64)
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        gripper_joint_pos = np.array(
            [self.gripper.get_current_position()], dtype=np.float64
        )
        wrench = np.array(state["ActualTCPForce"], dtype=np.float64)

        return {
            "joint_pos": np.concatenate((arm_joint_pos, gripper_joint_pos), dtype=np.float64),
            "joint_vel": np.concatenate((arm_joint_vel, np.zeros(1)), dtype=np.float64),
            "wrench": wrench,
        }

    def close(self):
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=2.0)
            self._sampler_thread = None
        try:
            self.arm.stop(wait=True)
        except Exception as exc:  # noqa: BLE001 - best-effort shutdown
            print(f"[{self.__class__.__name__}] Warning stopping arm: {exc}")
        if self._shm_manager is not None:
            try:
                self._shm_manager.shutdown()
            except Exception:
                pass
            self._shm_manager = None
        if hasattr(self.gripper, "disconnect"):
            try:
                self.gripper.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------ dry run ----

    def print_dry_run_summary(self, max_rows=20):
        if not self.dry_run:
            return
        commands = self.arm.commands
        print(f"\n[{self.__class__.__name__}] Dry-run summary")
        print(f"  arm commands: {len(commands)}, gripper commands: {len(self.gripper.commands)}")
        print(f"  sampler loop: {self.sampler_stats}")
        if not commands:
            return
        times = np.array([c[0] for c in commands])
        if len(times) > 1:
            gaps = np.diff(times) * 1e3
            print(
                f"  inter-command gap: mean {gaps.mean():.2f} ms "
                f"p95 {np.percentile(gaps, 95):.2f} ms max {gaps.max():.2f} ms "
                f"(target {self.spline_period * 1e3:.2f} ms)"
            )
        joints_all = np.array([c[1] for c in commands])
        travel = np.abs(joints_all.max(axis=0) - joints_all.min(axis=0))
        steps = np.abs(np.diff(joints_all, axis=0)) if len(joints_all) > 1 else np.zeros((1, 6))
        print(f"  joint range covered [rad]: {np.round(travel, 4)}")
        print(
            f"  per-tick step [rad]: max {steps.max():.5f} "
            f"(clamp {self.max_joint_step:.5f}, saturated on "
            f"{100.0 * (steps.max(axis=1) >= self.max_joint_step - 1e-9).mean():.1f}% of ticks)"
        )

        # Sample across the whole run, not just the head -- the first commands
        # are the pre-first-segment hold and say nothing about the policy.
        stride = max(1, len(commands) // max_rows)
        gripper_times = (
            np.array([g[0] for g in self.gripper.commands])
            if self.gripper.commands
            else None
        )
        print(f"  joint targets [rad] + gripper [cnt], every {stride} commands:")
        for t_cmd, joints, _ in commands[::stride][:max_rows]:
            if gripper_times is not None and len(gripper_times):
                grip = self.gripper.commands[
                    int(np.argmin(np.abs(gripper_times - t_cmd)))
                ][1]
            else:
                grip = float("nan")
            rel = t_cmd - commands[0][0]
            print(
                f"    t={rel:6.3f}s  "
                + " ".join(f"{q:+8.4f}" for q in joints)
                + f"  grip={grip:6.1f}"
            )
