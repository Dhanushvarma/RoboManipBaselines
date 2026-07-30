import argparse
import os
import time

import gymnasium as gym
import numpy as np
import yaml

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files
from robo_manip_baselines.policy.bspline_policy import BsplineAdapter as adapter
from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
    BSplineChunkFitter,
    ScipyBSplineCompression,
    bspline_span,
    whole_episode_params,
)

# Sibling module, not a package import: misc/ has no __init__.py and these are
# run as `python ./misc/Foo.py`, which puts misc/ on sys.path[0].
from ReplayErrorLog import append_row  # noqa: E402


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Replay a recorded demonstration at three levels of the B-spline "
            "stack, so a bad rollout can be blamed on one stage. Each mode adds "
            "exactly one component: raw (controller only) -> spline (+ the "
            "Algorithm 1 fit) -> segments (+ chunking, alignment and the "
            "high-rate sampler). bin/Rollout.py adds the trained network on top."
        ),
    )
    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="env configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="segments",
        choices=["raw", "spline", "segments"],
        help="raw: replay recorded commands. spline: fit the whole episode and "
        "replay the curve. segments: replay the chunked targets the policy "
        "consumes, through segment alignment and the sampler (oracle rollout)",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        nargs="*",
        default=None,
        help="which episodes to replay (default: all found)",
    )
    parser.add_argument(
        "--speedup",
        type=float,
        default=1.0,
        help="temporal rescaling; the demo is traversed this much faster",
    )
    parser.add_argument("--dry_run", action="store_true", help="no hardware")
    parser.add_argument(
        "--no_wait_before_start",
        action="store_true",
        help="skip the confirmation prompt before the arm moves",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save per-episode .npz replay logs",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="append error metrics to this CSV (created if absent). Runs at "
        "different --speedup accumulate into one file; plot it with "
        "PlotReplayErrors.py",
    )

    # ---- fit parameters: defaults MUST match training ----
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--bspline_degree", type=int, default=3)
    parser.add_argument(
        "--max_error",
        type=float,
        default=0.002,
        help="fit tolerance in weighted action units. Defaults to the TRAINING "
        "value, not the reference implementation's looser replay default, so "
        "this tests what the policy actually consumes",
    )
    parser.add_argument("--bspline_stride", type=int, default=1)
    parser.add_argument(
        "--gripper_weight", type=float, default=adapter.DEFAULT_GRIPPER_WEIGHT
    )
    parser.add_argument(
        "--gripper_action_idxes",
        type=int,
        nargs="*",
        default=list(adapter.DEFAULT_GRIPPER_ACTION_IDXES),
    )

    # ---- segments mode: replan cadence, mirroring RolloutBsplinePolicy ----
    parser.add_argument("--max_plan_age", type=float, default=0.4)
    parser.add_argument("--predict_before_end", type=float, default=0.1)
    parser.add_argument("--no_segment_align", action="store_true")

    return parser.parse_args()


class ReplayBsplineDemo:
    ENV_ID = "robo_manip_baselines/RealUR5eBSplineDemoEnv-v0"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.filenames = find_rmb_files(self.path)
        if self.episode_idx:
            self.filenames = [self.filenames[i] for i in self.episode_idx]

        with open(self.config, "r") as f:
            self.env_config = yaml.safe_load(f) or {}

        self.action_keys = [DataKey.COMMAND_JOINT_POS]
        self.weights = None
        self.env = None

    # ------------------------------------------------------------- setup ----

    def setup_env(self):
        config = dict(self.env_config)
        config["dry_run"] = self.dry_run
        # Replay never reads images, so the camera stack is dead weight and, on a
        # real robot, would slow the loop below the demo's own rate.
        config["camera_ids"] = {"hand": "dry-run"} if self.dry_run else None
        config.setdefault("pointcloud_camera_ids", None)
        config.setdefault("gelsight_ids", None)
        config.setdefault("sanwa_keyboard_ids", None)

        print(f"[{self.__class__.__name__}] Construct env: {self.ENV_ID}")
        self.env = gym.make(self.ENV_ID, **config).unwrapped

    def load_episode(self, filename):
        """Recorded commands, timestamps, and the measured trajectory."""
        with RmbData(filename) as rmb_data:
            time_seq = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
            command = np.asarray(
                rmb_data[DataKey.COMMAND_JOINT_POS][:], dtype=np.float64
            )
            measured = np.asarray(
                rmb_data[DataKey.MEASURED_JOINT_POS][:], dtype=np.float64
            )
        return time_seq, command, measured

    # ------------------------------------------------------------ replay ----

    def run(self):
        self.weights = adapter.build_action_weights(
            7, self.gripper_action_idxes, self.gripper_weight
        )
        time_info = adapter.measure_time_base(self.filenames)
        self.origin_time_scale = time_info["origin_time_scale"]

        self.setup_env()
        results = []
        try:
            for filename in self.filenames:
                results.append(self.replay_one(filename))
        finally:
            self.env.close()

        self.print_summary(results)

    def replay_one(self, filename):
        print(f"\n[{self.__class__.__name__}] {os.path.basename(filename)}")
        time_seq, command_seq, measured_seq = self.load_episode(filename)
        duration = time_seq[-1] - time_seq[0]
        print(
            f"[{self.__class__.__name__}]   {len(time_seq)} frames, "
            f"{duration:.2f} s at {self.origin_time_scale:.2f} Hz, "
            f"mode={self.mode}, speedup={self.speedup:.2f}x"
        )

        # Fit once up front; both spline modes need it and it is what the
        # policy would have been trained on.
        fitted_seq = None
        fitted_frame_idx = None
        params = None
        fitter = None
        if self.mode in ("spline", "segments"):
            weighted = command_seq * self.weights
            if self.mode == "spline":
                compressor = ScipyBSplineCompression(degree=self.bspline_degree)
                compressor.compress(weighted, max_error=self.max_error)
                if not compressor.hit_tolerance:
                    print(
                        f"[{self.__class__.__name__}]   WARNING: fit did not reach "
                        f"max_error={self.max_error}; results below are not "
                        f"representative of training targets"
                    )
                params = whole_episode_params(compressor)
                n_knots = len(params)
            else:
                fitter = BSplineChunkFitter(
                    [weighted],
                    chunk_size=self.chunk_size,
                    degree=self.bspline_degree,
                    max_error=self.max_error,
                    stride=self.bspline_stride,
                    verbose=False,
                )
                n_knots = fitter.n_knots_total
            fitted_seq, fitted_frame_idx = self.reconstruct(
                params, fitter, len(command_seq)
            )
            print(
                f"[{self.__class__.__name__}]   fit: {n_knots} knots, "
                f"compression {len(command_seq) / n_knots:.2f}x"
            )

        self.move_to_start(command_seq[0])

        if self.mode == "raw":
            log = self.replay_raw(time_seq, command_seq)
        elif self.mode == "spline":
            log = self.replay_spline(time_seq, params)
        else:
            log = self.replay_segments(time_seq, fitter)

        log.update(
            {
                "filename": os.path.basename(filename),
                "demo_command": command_seq,
                "demo_measured": measured_seq,
                "demo_time": time_seq,
                "fitted": fitted_seq if fitted_seq is not None else command_seq,
                "fitted_frame_idx": (
                    fitted_frame_idx
                    if fitted_frame_idx is not None
                    else np.arange(len(command_seq))
                ),
            }
        )
        metrics = self.report(log)
        self.save_log(filename, log)
        self.append_error_log(filename, metrics)
        return log

    def reconstruct(self, params, fitter, n_frames):
        """What the spline representation asserts, aligned to demo frames.

        Returns ``(fitted, frame_idx)``. ``frame_idx[i]`` is the demo frame that
        ``fitted[i]`` actually corresponds to -- normally ``i``, but a chunk is
        the *nearest future* segment, so its valid span can start a few frames
        ahead of the timestep it is anchored at. Comparing ``fitted[i]`` against
        ``demo[i]`` in that case measures the anchoring lead, not fit error.
        """
        from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
            eval_bspline_at,
        )

        if params is not None:
            values = eval_bspline_at(
                params, np.arange(n_frames), degree=self.bspline_degree
            )
            return np.asarray(values) / self.weights, np.arange(n_frames)

        fitted = np.zeros((n_frames, len(self.weights)))
        frame_idx = np.zeros(n_frames, dtype=int)
        for time_idx in range(n_frames):
            chunk = fitter.get_chunk(0, time_idx)
            t_min, t_max = bspline_span(chunk, degree=self.bspline_degree)
            # u = 0 is "this frame"; clamp into the chunk's valid span.
            offset = float(np.clip(0.0, t_min, t_max))
            fitted[time_idx] = (
                np.asarray(eval_bspline_at(chunk, offset, degree=self.bspline_degree))
                / self.weights
            )
            frame_idx[time_idx] = int(
                np.clip(round(time_idx + offset), 0, n_frames - 1)
            )
        return fitted, frame_idx

    def move_to_start(self, first_command):
        """Reset, then travel to the demo's first pose before replaying.

        ``_reset_robot`` goes to ``init_qpos``, which is not where the demo
        starts. Without this the first replay command would be a large jump.
        """
        self.env.stop_bspline()
        self.env._last_command = None
        self.env.reset()

        print(
            f"[{self.__class__.__name__}]   Moving to demo start pose "
            f"{np.round(first_command[:6], 3)}"
        )
        self.env._set_action(
            first_command, duration=None, joint_vel_limit_scale=0.3, wait=True
        )
        self.env._last_command = np.asarray(first_command, dtype=np.float64).copy()
        # Refresh arm_joint_pos_actual, which overwrite_command_for_safety uses
        # as "where the arm is". Normally env.step() does this; the replay loop
        # bypasses step(), so without it the safety check still believes the arm
        # is at init_qpos and rejects the first replay command as a large jump.
        self.env._get_obs()

        if not self.no_wait_before_start and not self.dry_run:
            input(f"[{self.__class__.__name__}]   Press Enter to start replay. ")

    # ------------------------------------------------------------- modes ----

    def replay_raw(self, time_seq, command_seq):
        """Command each recorded frame, paced by the recorded timestamps.

        Deliberately not ``env.step()``: that paces at ``env.dt`` (50 Hz), which
        would play a 30 Hz demo 1.67x too fast.
        """
        rel_time = (time_seq - time_seq[0]) / self.speedup
        commanded, measured, stamps = [], [], []

        start = time.perf_counter()
        for time_idx, target_time in enumerate(rel_time):
            self.sleep_until(start + target_time)
            period = (
                rel_time[time_idx + 1] - target_time
                if time_idx + 1 < len(rel_time)
                else rel_time[-1] - rel_time[-2]
            )
            self.env._set_action(
                command_seq[time_idx].copy(), duration=max(period, 1e-3), wait=False
            )
            commanded.append(np.asarray(self.env._last_command).copy())
            measured.append(self.env._get_obs()["joint_pos"].copy())
            stamps.append(time.perf_counter() - start)

        return {
            "commanded": np.array(commanded),
            "measured": np.array(measured),
            "stamps": np.array(stamps),
            "wall_duration": time.perf_counter() - start,
            "n_segments": 0,
        }

    def replay_spline(self, time_seq, params):
        """Install the whole episode as one segment and let the sampler drive."""
        self.env.install_bspline_segment(
            params,
            origin_time_scale=self.origin_time_scale,
            speedup=self.speedup,
            degree=self.bspline_degree,
            weights=self.weights,
            align=False,  # nothing to align to; start at the beginning
        )
        return self.follow(time_seq, n_segments=1)

    def replay_segments(self, time_seq, fitter):
        """Oracle rollout: install ground-truth chunks at the replan cadence."""
        n_frames = len(time_seq)
        rel_time = (time_seq - time_seq[0]) / self.speedup
        commanded, measured, stamps = [], [], []
        n_segments = 0

        start = time.perf_counter()
        for time_idx, target_time in enumerate(rel_time):
            self.sleep_until(start + target_time)

            if self.needs_replan():
                # The frame the policy would be conditioned on right now.
                elapsed = time.perf_counter() - start
                anchor = int(
                    np.clip(
                        round(elapsed * self.speedup * self.origin_time_scale),
                        0,
                        n_frames - 1,
                    )
                )
                installed = self.env.install_bspline_segment(
                    fitter.get_chunk(0, anchor).astype(np.float64),
                    origin_time_scale=self.origin_time_scale,
                    speedup=self.speedup,
                    degree=self.bspline_degree,
                    weights=self.weights,
                    align=not self.no_segment_align,
                )
                n_segments += int(bool(installed))

            commanded.append(self.current_command())
            measured.append(self.env._get_obs()["joint_pos"].copy())
            stamps.append(time.perf_counter() - start)

        self.env.stop_bspline()
        return {
            "commanded": np.array(commanded),
            "measured": np.array(measured),
            "stamps": np.array(stamps),
            "wall_duration": time.perf_counter() - start,
            "n_segments": n_segments,
        }

    def follow(self, time_seq, n_segments):
        """Sample commanded/measured at demo frame times while the sampler runs."""
        rel_time = (time_seq - time_seq[0]) / self.speedup
        commanded, measured, stamps = [], [], []

        start = time.perf_counter()
        for target_time in rel_time:
            self.sleep_until(start + target_time)
            commanded.append(self.current_command())
            measured.append(self.env._get_obs()["joint_pos"].copy())
            stamps.append(time.perf_counter() - start)

        self.env.stop_bspline()
        return {
            "commanded": np.array(commanded),
            "measured": np.array(measured),
            "stamps": np.array(stamps),
            "wall_duration": time.perf_counter() - start,
            "n_segments": n_segments,
        }

    def needs_replan(self):
        state = self.env.get_segment_state()
        if state is None:
            return True
        if state["seconds_remaining"] < self.predict_before_end * self.speedup:
            return True
        return self.max_plan_age > 0 and state["age"] > self.max_plan_age

    def current_command(self):
        last = self.env._last_command
        if last is None:
            return np.zeros(len(self.weights))
        return np.asarray(last, dtype=np.float64).copy()

    @staticmethod
    def sleep_until(deadline):
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------ report ----

    def report(self, log):
        """Decompose demo -> [fit] -> commanded -> [tracking] -> measured."""
        name = self.__class__.__name__
        demo = log["demo_command"]
        fitted = log["fitted"]
        commanded = log["commanded"]
        measured = log["measured"]
        frame_idx = log["fitted_frame_idx"]
        n = min(len(demo), len(commanded), len(measured))
        # The fit row compares each fitted sample against the demo frame it
        # actually represents (see reconstruct).
        demo_for_fit = demo[frame_idx[:n]]
        lead = frame_idx[:n] - np.arange(n)

        def stat(a, b, sl, scale=1.0):
            err = np.abs(a[:n, sl] - b[:n, sl]) * scale
            return err.max(), float(np.sqrt(np.mean(err**2)))

        rad2deg = np.rad2deg(1.0)
        # Each row compares like with like. "fit" is frame-aligned (a chunk can
        # lead its anchor); the other two compare at matched wall time. "execute"
        # is measured against the demo, not against "fitted", so it nests: it is
        # fit error plus whatever chunk stitching and the clamp add on top.
        rows = [
            ("fit", "fit      demo -> fitted   (representation)", fitted, demo_for_fit),
            (
                "execute",
                "execute  demo -> commanded (+ stitching, clamp)",
                commanded,
                demo,
            ),
            ("track", "track    commanded -> measured (controller)", measured, commanded),
        ]
        print(f"[{name}]   error decomposition (n={n} frames):")
        print(
            f"[{name}]     {'stage':<48} {'joint max':>10} {'joint RMS':>10} "
            f"{'grip max':>9}"
        )
        metrics = {"n_frames": n}
        for key, label, a, b in rows:
            j_max, j_rms = stat(a, b, slice(0, 6), rad2deg)
            g_max, _ = stat(a, b, slice(6, 7))
            metrics[f"{key}_joint_max_deg"] = j_max
            metrics[f"{key}_joint_rms_deg"] = j_rms
            metrics[f"{key}_grip_max_cnt"] = g_max
            print(
                f"[{name}]     {label:<48} {j_max:9.4f}d {j_rms:9.4f}d {g_max:8.2f}c"
            )
        metrics["anchor_lead_max"] = int(lead.max())

        if lead.max() > 0:
            print(
                f"[{name}]     (chunk anchoring lead: median {int(np.median(lead))}"
                f" frames, max {int(lead.max())} -- the nearest future segment "
                f"can start ahead of its anchor)"
            )

        wall = log["wall_duration"]
        expected = (log["demo_time"][-1] - log["demo_time"][0]) / self.speedup
        print(
            f"[{name}]   timing: {wall:.2f} s wall vs {expected:.2f} s expected "
            f"({100.0 * (wall / expected - 1.0):+.1f}%)"
        )
        if log["n_segments"]:
            print(f"[{name}]   segments installed: {log['n_segments']}")
        print(f"[{name}]   sampler: {self.env.sampler_stats}")
        if self.dry_run:
            print(
                f"[{name}]   NOTE: dry run -- 'track' is meaningless (the stub "
                f"echoes commands back). Only 'fit' and 'execute' are real."
            )

        metrics.update(
            {
                "wall_duration_s": wall,
                "expected_duration_s": expected,
                "n_segments": log["n_segments"],
            }
        )
        return metrics

    def append_error_log(self, filename, metrics):
        if self.log is None:
            return
        row = dict(metrics)
        row.update(
            {
                "script": self.__class__.__name__,
                "mode": self.mode,
                "episode": os.path.splitext(os.path.basename(filename))[0],
                "speedup": self.speedup,
                "max_error": self.max_error,
                "chunk_size": self.chunk_size,
                "max_plan_age": self.max_plan_age,
            }
        )
        append_row(self.log, row)
        print(f"[{self.__class__.__name__}]   Logged metrics: {self.log}")

    def save_log(self, filename, log):
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filename))[0]
        out_path = os.path.join(self.output_dir, f"{stem}_{self.mode}.npz")
        np.savez(
            out_path, **{k: v for k, v in log.items() if not isinstance(v, str)}
        )
        print(f"[{self.__class__.__name__}]   Saved log: {out_path}")

    def print_summary(self, results):
        if len(results) <= 1:
            return
        name = self.__class__.__name__
        print(f"\n[{name}] {len(results)} episodes, mode={self.mode}")
        for key, sl, scale, unit in (
            ("joint", slice(0, 6), np.rad2deg(1.0), "deg"),
            ("gripper", slice(6, 7), 1.0, "cnt"),
        ):
            fit = [
                np.abs(
                    r["fitted"][:, sl]
                    - r["demo_command"][r["fitted_frame_idx"]][:, sl]
                ).max()
                * scale
                for r in results
            ]
            track = [
                np.abs(r["measured"][:, sl] - r["commanded"][:, sl]).max() * scale
                for r in results
            ]
            print(
                f"[{name}]   {key:<8} fit max {np.max(fit):8.4f} {unit}   "
                f"track max {np.max(track):8.4f} {unit}"
            )


if __name__ == "__main__":
    ReplayBsplineDemo(**vars(parse_argument())).run()
