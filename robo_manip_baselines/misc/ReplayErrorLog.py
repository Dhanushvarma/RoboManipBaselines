"""Append-only CSV log of replay error metrics, shared by the replay scripts.

One row per (episode, mode, speedup). Append-only on purpose: a speedup sweep is
several separate process invocations, and they all accumulate into one file that
``PlotReplayErrors.py`` reads.

Both ``ReplayBsplineDemo.py`` and ``ReplayRealUR5eDemo.py`` write this schema, so
a single plot can put the ur_rtde baseline on the same axes as the B-spline modes.
Columns a given script cannot measure are left empty rather than zero -- the
plotter skips blanks, and a zero would read as "no error".
"""

import csv
import os
from datetime import datetime

FIELDS = [
    "timestamp",
    "script",
    "mode",
    "episode",
    "speedup",
    "n_frames",
    # Joint error in degrees, per stage.
    "fit_joint_max_deg",
    "fit_joint_rms_deg",
    "execute_joint_max_deg",
    "execute_joint_rms_deg",
    "track_joint_max_deg",
    "track_joint_rms_deg",
    # Gripper in counts. Logged even though the current plots ignore it, so a
    # sweep never has to be repeated just to get these back.
    "fit_grip_max_cnt",
    "execute_grip_max_cnt",
    "track_grip_max_cnt",
    # Reference: how well the arm tracked when the data was recorded. Only
    # ReplayRealUR5eDemo computes it; it is the yardstick for the track stage.
    "recorded_joint_max_deg",
    "recorded_joint_rms_deg",
    # Context needed to interpret a row.
    "wall_duration_s",
    "expected_duration_s",
    "max_error",
    "chunk_size",
    "max_plan_age",
    "n_segments",
    "anchor_lead_max",
]


def append_row(log_path, row):
    """Append one row, writing the header if the file is new."""
    directory = os.path.dirname(os.path.abspath(log_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    payload = {key: row.get(key, "") for key in FIELDS}
    payload["timestamp"] = payload["timestamp"] or datetime.now().isoformat(
        timespec="seconds"
    )

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(payload)

    return log_path
