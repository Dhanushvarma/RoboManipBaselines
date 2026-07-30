import argparse
import os
import time

import gymnasium as gym
import numpy as np
import yaml

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Replay recorded demonstrations verbatim through RealUR5eEnvBase "
            "(ur_rtde servoJ) and report joint tracking error. No B-spline, no "
            "policy, no checkpoint. This is the control baseline: the demos were "
            "recorded through this same env, so anything wrong here is the robot "
            "or the controller, not the action representation."
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
        help="replay this much faster than recorded. Above 1.0 the commanded "
        "joint speed scales with it, so raise carefully",
    )
    parser.add_argument(
        "--joint_vel_limit_scale",
        type=float,
        default=2.0,
        help="passed to overwrite_command_for_safety; RealEnvBase.step uses 2.0",
    )
    parser.add_argument(
        "--no_wait_before_start",
        action="store_true",
        help="skip the confirmation prompt before the arm moves",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to save per-episode .npz logs",
    )
    return parser.parse_args()


class ReplayRealUR5eDemo:
    ENV_ID = "robo_manip_baselines/RealUR5eDemoEnv-v0"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.filenames = find_rmb_files(self.path)
        if self.episode_idx:
            self.filenames = [self.filenames[i] for i in self.episode_idx]

        with open(self.config, "r") as f:
            self.env_config = yaml.safe_load(f) or {}

        self.env = None

    # -------------------------------------------------------------- setup ---

    def setup_env(self):
        config = dict(self.env_config)
        # Replay reads no images; skipping the cameras keeps the loop able to
        # hold the demo's own rate.
        config["camera_ids"] = None
        config.setdefault("pointcloud_camera_ids", None)
        config.setdefault("gelsight_ids", None)
        config.setdefault("sanwa_keyboard_ids", None)

        print(f"[{self.__class__.__name__}] Construct env: {self.ENV_ID}")
        self.env = gym.make(self.ENV_ID, **config).unwrapped

    # ------------------------------------------------------------- replay ---

    def run(self):
        self.setup_env()
        results = []
        try:
            for filename in self.filenames:
                results.append(self.replay_one(filename))
        finally:
            self.env.close()
        self.print_summary(results)

    def replay_one(self, filename):
        name = self.__class__.__name__
        print(f"\n[{name}] {os.path.basename(filename)}")

        with RmbData(filename) as rmb_data:
            time_seq = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
            command_seq = np.asarray(
                rmb_data[DataKey.COMMAND_JOINT_POS][:], dtype=np.float64
            )
            recorded_measured = np.asarray(
                rmb_data[DataKey.MEASURED_JOINT_POS][:], dtype=np.float64
            )

        rate = 1.0 / np.median(np.diff(time_seq))
        print(
            f"[{name}]   {len(time_seq)} frames, "
            f"{time_seq[-1] - time_seq[0]:.2f} s at {rate:.2f} Hz, "
            f"speedup={self.speedup:.2f}x"
        )

        self.move_to_start(command_seq[0])

        # Pace from the recorded timestamps, not env.dt: RealEnvBase.step would
        # hold 50 Hz, which plays a 30 Hz demo 1.67x too fast.
        rel_time = (time_seq - time_seq[0]) / self.speedup
        commanded, measured, stamps = [], [], []

        start = time.perf_counter()
        for time_idx, target_time in enumerate(rel_time):
            remaining = start + target_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

            if time_idx + 1 < len(rel_time):
                period = rel_time[time_idx + 1] - target_time
            else:
                period = rel_time[-1] - rel_time[-2]

            action = command_seq[time_idx].copy()
            self.env._set_action(
                action,
                duration=max(period, 1e-3),
                joint_vel_limit_scale=self.joint_vel_limit_scale,
                wait=False,
            )
            obs = self.env._get_obs()
            # _set_action may clamp the command for safety, so log what it
            # became rather than what was asked for.
            commanded.append(action.copy())
            measured.append(obs["joint_pos"].copy())
            stamps.append(time.perf_counter() - start)

        log = {
            "filename": os.path.basename(filename),
            "demo_command": command_seq,
            "recorded_measured": recorded_measured,
            "commanded": np.array(commanded),
            "measured": np.array(measured),
            "stamps": np.array(stamps),
            "wall_duration": time.perf_counter() - start,
            "demo_duration": (time_seq[-1] - time_seq[0]) / self.speedup,
        }
        self.report(log)
        self.save_log(filename, log)
        return log

    def move_to_start(self, first_command):
        name = self.__class__.__name__
        self.env.reset()
        print(
            f"[{name}]   Moving to demo start pose {np.round(first_command[:6], 3)}"
        )
        self.env._set_action(
            first_command, duration=None, joint_vel_limit_scale=0.3, wait=True
        )
        # Refresh arm_joint_pos_actual, which overwrite_command_for_safety reads
        # as "where the arm is". env.step() normally does this; this loop calls
        # _set_action directly, so without it the safety check would still think
        # the arm sits at init_qpos.
        self.env._get_obs()

        if not self.no_wait_before_start:
            input(f"[{name}]   Press Enter to start replay. ")

    # ------------------------------------------------------------- report ---

    def report(self, log):
        name = self.__class__.__name__
        commanded = log["commanded"]
        measured = log["measured"]
        recorded = log["recorded_measured"]
        n = min(len(commanded), len(measured), len(recorded))

        def stat(a, b, sl, scale=1.0):
            err = np.abs(a[:n, sl] - b[:n, sl]) * scale
            return err.max(), float(np.sqrt(np.mean(err**2)))

        deg = np.rad2deg(1.0)
        # The first row is computed entirely from the file: how well the robot
        # tracked at recording time. It is the benchmark the other two are judged
        # against -- a replay that tracks no worse than the recording did is
        # behaving correctly, however large the absolute number looks.
        rows = [
            ("recorded demo_command -> recorded_measured", recorded, log["demo_command"]),
            ("track    commanded -> measured (this run)", measured, commanded),
            ("repeat   recorded_measured -> this run's", measured, recorded),
        ]
        print(f"[{name}]   tracking (n={n} frames):")
        print(
            f"[{name}]     {'stage':<46} {'joint max':>10} {'joint RMS':>10} "
            f"{'grip max':>9}"
        )
        for label, a, b in rows:
            j_max, j_rms = stat(a, b, slice(0, 6), deg)
            g_max, _ = stat(a, b, slice(6, 7))
            print(
                f"[{name}]     {label:<46} {j_max:9.4f}d {j_rms:9.4f}d {g_max:8.2f}c"
            )

        per_joint = np.rad2deg(
            np.abs(measured[:n, :6] - commanded[:n, :6])
        ).max(axis=0)
        print(
            f"[{name}]     per-joint track max [deg]: "
            + "  ".join(f"j{j}={per_joint[j]:.3f}" for j in range(6))
        )

        wall = log["wall_duration"]
        expected = log["demo_duration"]
        print(
            f"[{name}]   timing: {wall:.2f} s wall vs {expected:.2f} s expected "
            f"({100.0 * (wall / expected - 1.0):+.1f}%)"
        )

    def save_log(self, filename, log):
        if self.output_dir is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(filename))[0]
        out_path = os.path.join(self.output_dir, f"{stem}_raw_replay.npz")
        np.savez(
            out_path, **{k: v for k, v in log.items() if not isinstance(v, str)}
        )
        print(f"[{self.__class__.__name__}]   Saved log: {out_path}")

    def print_summary(self, results):
        if len(results) <= 1:
            return
        name = self.__class__.__name__
        deg = np.rad2deg(1.0)
        track = [
            np.abs(r["measured"][:, :6] - r["commanded"][:, :6]).max() * deg
            for r in results
        ]
        grip = [
            np.abs(r["measured"][:, 6] - r["commanded"][:, 6]).max() for r in results
        ]
        print(f"\n[{name}] {len(results)} episodes")
        print(
            f"[{name}]   joint track max {np.max(track):.4f} deg "
            f"(median {np.median(track):.4f})"
        )
        print(
            f"[{name}]   gripper track max {np.max(grip):.2f} cnt "
            f"(median {np.median(grip):.2f})"
        )


if __name__ == "__main__":
    ReplayRealUR5eDemo(**vars(parse_argument())).run()
