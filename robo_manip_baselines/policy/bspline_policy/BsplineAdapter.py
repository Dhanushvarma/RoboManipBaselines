"""The RMB <-> B-spline Policy seam.

Everything RMB-specific about feeding Algorithm 1 lives here, so
:mod:`BSplineAction` can stay a faithful port of the reference implementation.

Three jobs:

1. **Read RMB episodes** into the per-episode ``(T, D)`` arrays the fitter wants.
2. **Per-channel weighting.** RMB packs arm and gripper into one
   ``command_joint_pos`` vector -- joints in radians, gripper in Robotiq counts
   0-255. A single scalar ``max_error`` across those units is dominated by the
   gripper, so each channel is scaled by ``w`` before fitting and the control
   points are divided by ``w`` after decode. Because a B-spline is linear in its
   control points and the least-squares decouples per channel, this is *exact*:
   the only thing ``w`` changes is which knots the adaptive fitter picks.
   scipy has no per-channel weight (``make_lsq_spline(w=...)`` is per-sample),
   so pre-scaling is the only way to express one.
3. **Time base.** The fit runs in frame-index time, so the knots the policy
   predicts are in frames. ``origin_time_scale`` is the frames-per-second factor
   that converts them to wall-clock at rollout, and it must equal the rate the
   fit saw. It is measured from ``DataKey.TIME`` rather than assumed, because
   the nominal ``RealEnvBase.dt = 0.02`` is not what the recorder achieves
   (measured: ~30 Hz, because the camera is read synchronously).
"""

import os

import numpy as np

from robo_manip_baselines.common import DataKey, RmbData, convert_data_to_policy

# Default per-channel weight for a Robotiq gripper channel expressed in raw
# counts. 1/1000 pairs with eps=0.002 to give a 2-count tolerance, which is
# ~0.39 mm on a Hand-e (50 mm stroke) and ~0.67 mm on a 2F-85 (85 mm) -- below
# anything that affects a grasp, while not spending knots on precision the
# hardware cannot express.
DEFAULT_GRIPPER_WEIGHT = 1.0e-3

# UR5e single-arm layout: 6 arm joints then 1 gripper channel.
DEFAULT_GRIPPER_ACTION_IDXES = (6,)


def build_action_weights(action_dim, gripper_action_idxes, gripper_weight):
    """Per-channel weight vector: 1.0 for joints, ``gripper_weight`` for grippers."""
    weights = np.ones(action_dim, dtype=np.float64)
    for idx in gripper_action_idxes:
        if not (0 <= idx < action_dim):
            raise ValueError(
                f"[BsplineAdapter] gripper action index {idx} out of range for "
                f"action_dim {action_dim}"
            )
        weights[idx] = float(gripper_weight)
    return weights


def measure_time_base(filenames, verbose=True):
    """Measure the recording rate and its jitter from ``DataKey.TIME``.

    Returns a dict with ``origin_time_scale`` (Hz, the number the rollout needs),
    ``median_dt``, and jitter statistics. A wrong ``origin_time_scale`` is a
    silent time-scaling error -- the robot traces the right path at the wrong
    speed -- so the jitter numbers are reported loudly rather than assumed away.
    """
    all_dt = []
    per_file = []
    for filename in filenames:
        with RmbData(filename) as rmb_data:
            time_seq = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
        if len(time_seq) < 2:
            continue
        dt = np.diff(time_seq)
        all_dt.append(dt)
        per_file.append((filename, float(np.median(dt)), len(time_seq)))

    if not all_dt:
        raise ValueError("[BsplineAdapter] No usable time data in the dataset.")

    all_dt = np.concatenate(all_dt)
    median_dt = float(np.median(all_dt))
    rel = np.abs(all_dt / median_dt - 1.0)
    info = {
        "origin_time_scale": 1.0 / median_dt,
        "median_dt": median_dt,
        "jitter_p95": float(np.percentile(rel, 95)),
        "jitter_max": float(rel.max()),
        "num_episodes": len(per_file),
    }

    if verbose:
        print(
            f"[BsplineAdapter] Time base: {info['origin_time_scale']:.2f} Hz "
            f"(median dt {median_dt * 1e3:.2f} ms), "
            f"jitter p95 {info['jitter_p95'] * 100:.1f}% max {info['jitter_max'] * 100:.1f}%"
        )
        if info["jitter_p95"] > 0.10:
            print(
                "[BsplineAdapter] WARNING: timing jitter above 10%. The B-spline "
                "fit assumes a uniform dt (it fits in frame-index time), so a "
                "jittery dataset is silently time-distorted. Consider "
                "resampling onto a uniform grid before training."
            )

    return info


def load_episode_actions(filenames, action_keys, weights=None, verbose=True):
    """Load one ``(T, D)`` action array per episode, optionally weighted.

    Actions are read at **full rate**, deliberately ignoring ``--skip``. The
    adaptive fitter places knots by trajectory curvature, which is a property of
    the motion rather than the sampling rate, so subsampling barely reduces the
    knot count while cutting the sample count proportionally -- measured on the
    UR5e data, ``--skip 3`` dropped compression from 5.41x to 1.85x for no
    benefit. ``--skip`` still governs observation windowing in the dataset.
    """
    episode_actions_list = []
    for filename in filenames:
        with RmbData(filename) as rmb_data:
            action = np.concatenate(
                [
                    convert_data_to_policy(np.asarray(rmb_data[key][:]), key)
                    for key in action_keys
                ],
                axis=1,
            ).astype(np.float64)
        if weights is not None:
            action = action * weights
        episode_actions_list.append(action)

    if verbose:
        lengths = [len(a) for a in episode_actions_list]
        print(
            f"[BsplineAdapter] Loaded {len(lengths)} episodes, "
            f"{sum(lengths)} samples, action_dim {episode_actions_list[0].shape[1]}, "
            f"lengths {min(lengths)}-{max(lengths)}"
        )

    return episode_actions_list


def build_fitter(filenames, action_keys, bspline_info, cache_dir=None, verbose=True):
    """Load episodes and fit them, reusing a cached fit when one matches.

    Single entry point for both ``TrainBsplinePolicy.set_data_stats`` and
    ``BsplinePolicyDataset.setup_variables``, which otherwise refit the same
    episodes on every training start (three times over: stats, train split, val
    split). The cache key hashes the file list and every parameter that changes
    the fit, so editing any of them invalidates it automatically -- the train
    and val splits get different keys because their file lists differ.
    """
    from .BSplineAction import BSplineChunkFitter, make_bspline_cache_path

    weights = np.asarray(bspline_info["action_weights"], dtype=np.float64)

    cache_path = None
    if cache_dir is not None:
        cache_path = make_bspline_cache_path(
            cache_dir,
            filenames,
            chunk_size=bspline_info["chunk_size"],
            degree=bspline_info["degree"],
            max_error=bspline_info["max_error"],
            stride=bspline_info["stride"],
            weights=weights,
            relative_knots=bspline_info["relative_knots"],
        )

    # Skip the episode load entirely on a cache hit -- it is the expensive half
    # for a large dataset, and the fitter does not need it.
    if cache_path is not None and os.path.exists(cache_path):
        episode_actions_list = None
    else:
        episode_actions_list = load_episode_actions(
            filenames, action_keys, weights=weights, verbose=verbose
        )

    return BSplineChunkFitter(
        episode_actions_list,
        chunk_size=bspline_info["chunk_size"],
        degree=bspline_info["degree"],
        max_error=bspline_info["max_error"],
        stride=bspline_info["stride"],
        relative_knots=bspline_info["relative_knots"],
        cache_path=cache_path,
        verbose=verbose,
    )


def build_action_stats(fitter, norm_type, out_min=-1.0, out_max=1.0):
    """Build ``model_meta_info["action"]`` stats for a B-spline parameter matrix.

    The reference implementation reduces its stats over the **row** axis and
    broadcasts back, giving one scale per channel shared by all rows. That is
    not a stylistic choice: a knot vector is valid only if non-decreasing, and a
    single positive affine map applied uniformly to every row preserves that
    (``u_i <= u_j`` implies ``s*u_i + o <= s*u_j + o`` for ``s > 0``).
    Per-element stats -- which is what RMB's ``calc_stats_from_seq`` would
    produce -- give each row its own scale, and monotonicity would not survive
    the round trip. So this reproduces the reference's per-channel reduction.

    Stats come out shaped ``(n_action_steps, n_action_channels)`` and
    ``normalize_data`` / ``denormalize_data`` broadcast over them directly.
    """
    raw = fitter.get_action_stats()  # each (1, L, C)
    n_steps, n_channels = raw["min"].shape[1], raw["min"].shape[2]

    per_channel = {
        "min": np.min(raw["min"], axis=1, keepdims=True),
        "max": np.max(raw["max"], axis=1, keepdims=True),
        "mean": np.mean(raw["mean"], axis=1, keepdims=True),
        "std": np.mean(raw["std"], axis=1, keepdims=True),
    }

    stats = {
        key: np.ascontiguousarray(
            np.broadcast_to(value[0], (n_steps, n_channels)), dtype=np.float64
        )
        for key, value in per_channel.items()
    }

    # Same guards RMB's calc_stats_from_seq applies. These also make a constant
    # channel (e.g. a gripper that never moved in a demo) safe rather than a
    # divide-by-zero.
    stats["range"] = np.clip(stats["max"] - stats["min"], 1e-3, 1e10)
    stats["std"] = np.clip(stats["std"], 1e-3, 1e10)

    norm_config = {"type": norm_type}
    if norm_type == "limits":
        norm_config["out_min"] = out_min
        norm_config["out_max"] = out_max

    stats["norm_config"] = norm_config
    stats["example"] = fitter.all_actions[0].astype(np.float64)

    if not np.all(np.isfinite(stats["range"])) or not np.all(np.isfinite(stats["std"])):
        raise ValueError("[BsplineAdapter] Non-finite action stats.")

    return stats


def unweight_actions(actions, weights):
    """Undo the per-channel weighting on decoded actions (``(..., D)``)."""
    return np.asarray(actions, dtype=np.float64) / weights


def unweight_control_points(action_params, weights):
    """Undo the per-channel weighting on a ``(L, 1 + D)`` parameter matrix.

    Column 0 (knots) is left alone -- it is in frames, not action units.
    """
    params = np.array(action_params, dtype=np.float64, copy=True)
    params[..., 1:] = params[..., 1:] / weights
    return params
