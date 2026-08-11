# UMI per-key latency alignment — reference

Status: **not implemented in `UmiDp`.** This document explains what the feature is, how UMI implements
it, and what porting it into RMB would require. Written as a reference for whoever picks it up.

## The problem

Sensors report the past, and they do so by different amounts. A camera frame that *arrives* at time `t`
depicts the world at `t − 0.125 s`; a robot joint reading that arrives at `t` depicts `t − 0.0001 s`.

If both are written to index `i` of a recording, the image and the proprioception at that index describe
the world **at different moments**. A policy trained on that pairing learns a systematically skewed
correspondence between what it sees and where the arm is.

This matters more for a *relative* action space than an absolute one, because the measured end-effector
pose is the **anchor**: every predicted action is expressed in a frame derived from a pose that does not
correspond to the image the policy is looking at. The error enters every action in the chunk, not just
the state input.

At RMB's ~23.4 Hz (`dt` ≈ 42.7 ms), a 125 ms camera lag is about **3 timesteps**.

## How UMI represents it

Per-key constants in `third_party/universal_manipulation_interface/diffusion_policy/config/task/umi.yaml`:

```yaml
camera_obs_latency: 0.125     # seconds
robot_obs_latency:  0.0001
gripper_obs_latency: 0.02
dataset_frequeny: 0 #59.94    # sic; set to your dataset's sample rate
```

Each observation key then declares a `latency_steps`, a **fractional** index offset:

```yaml
camera0_rgb:            latency_steps: 0                                              # umi.yaml:20
robot0_eef_pos:         latency_steps: (camera_obs_latency - robot_obs_latency)   * f # umi.yaml:27
robot0_gripper_width:   latency_steps: (camera_obs_latency - gripper_obs_latency) * f # umi.yaml:57
```

Two things follow from the formula:

- **The camera is the reference clock.** Every offset is measured *relative to* camera latency, so
  `camera0_rgb` is 0 by construction, and `sampler.py:135` asserts it. Other sensors are resampled onto
  the camera's timeline, not the other way round.
- **`dataset_frequeny` converts seconds to steps**, so it must be set to the recording rate. Left at `0`,
  every `latency_steps` evaluates to `0.0` and the feature is inert — which is the state of the shipped
  config.

Actions carry no latency: `sampler.py:189` asserts `action_latency_steps == 0`.

## How UMI applies it

`diffusion_policy/common/sampler.py:128-176`. Because `latency_steps` is fractional, the offset sample
does not exist in the array and must be **interpolated**.

```python
# sampler.py:147-153
idx_with_latency = np.array(
    [current_idx - idx * this_downsample_steps + this_latency_steps
     for idx in range(this_horizon)], dtype=np.float32)
idx_with_latency = idx_with_latency[::-1]
idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)   # +/-5 sample guard band
```

Then, by key type:

| Key | Method | Line |
| --- | --- | --- |
| rgb | none — integer slice, `latency_steps` asserted 0 | 135-145 |
| `*_rot_*` | `scipy.spatial.transform.Slerp` over quaternions or rotvecs | 155-169 |
| everything else | `scipy.interpolate.interp1d`, linear | 171-175 |

The rotation branch is the part most likely to be got wrong. Rotations are converted **out** of their
storage form (`from_quat` / `from_rotvec`), slerped, then converted back — never interpolated
component-wise.

## Porting into RMB

### Training — `policy/umi_dp/UmiDpDataset.py`

Roughly 30–40 lines.

1. Add `--camera_obs_latency`, `--robot_obs_latency`, `--gripper_obs_latency` (seconds) and derive the
   rate from the data (`rmb_data[DataKey.TIME]`), rather than asking the user for `dataset_frequeny`.
2. `latency_steps = (camera_latency − sensor_latency) × rate`, in **raw** index units.
3. Interpolate the low-dim keys at the fractional indices; leave images on integer indices.

**The trap:** interpolate *before* `get_pose9_from_pose7`, not after. RMB stores poses as
`(x, y, z, qw, qx, qy, qz)`, so slerp the quaternion and then derive the 6D rotation. Interpolating the
6D rotation component-wise produces a non-rotation, which `_get_matrix_from_rotation_6d` will silently
re-orthonormalize into something wrong.

Note RMB applies `skip` when loading. Simplest correct approach: interpolate on the **unskipped** arrays
at raw fractional indices, then take the skipped positions — do not mix skipped indices with raw
`latency_steps`.

### Rollout — `policy/umi_dp/RolloutUmiDp.py`

Roughly 30 lines, and conceptually the harder half: at inference you cannot interpolate into the future.

UMI solves this in `umi/real_world/`, a real-time stack that timestamps every observation and resamples
onto a common clock. **RMB has no equivalent** — `RolloutBase` receives one observation dict per step,
and the `.rmb` format stores a single `time` array with no per-sensor timestamps.

The tractable version, symmetric with training and requiring no real timestamps:

- keep a proprioception ring buffer deeper than `obs_horizon` by `ceil(latency_steps) + 1`;
- when assembling the policy input, sample that buffer at `now − latency_steps` by interpolation;
- keep using the *configured* constant and the loop rate, exactly as training does.

This is causal and needs only a deeper buffer.

### Verification

`RolloutUmiDp`'s observation window must stay bit-identical to `UmiDpDataset`'s. There is already a check
that reports `0.000e+00` across all four observation keys; re-run it with a non-zero latency. If the two
diverge, training and rollout are anchored differently, which is the failure mode this whole subsystem
exists to prevent.

## Measure before implementing

The code is the easy part. The number is not, and a wrong constant is **worse than zero** — it shifts
proprioception away from truth instead of toward it. UMI's 125 ms was a GoPro; a USB camera will differ.

Two ways to measure, both usable offline on existing recordings:

- Command a sharp end-effector motion, cross-correlate commanded velocity against image-derived motion
  (optical flow or frame differencing), and take the lag at peak correlation.
- Flash an LED on a known trigger and compare the trigger timestamp against its appearance in frame.

Decision rule: at ~23.4 Hz, if the measured lag is **under one step (43 ms)** the feature is noise and
not worth building. **Two to three steps** justifies the work.
