# B-spline Policy in RoboManipBaselines

BSP replaces discrete action chunks with a continuous B-spline curve. The
network predicts spline *parameters*; the curve is sampled at execution time, so
the controller gets smooth high-rate commands from a slow policy, and the same
curve can be retimed for faster execution without retraining.

This integration targets **real UR5e, absolute joint positions, via
`callm_controller`**.

## How it works

```
train:   demo (T,7) --fit--> knots + control points --chunk--> (16,8) targets --> Diffusion U-Net
rollout: obs --> (16,8) params --align--> segment --100 Hz sampler--> servoJ
```

**Representation.** One target is a `(chunk_size + 2*degree, 1 + action_dim)`
matrix = `(16, 8)`. Column 0 is the knot vector; columns 1: are control points.
Knots are in **dataset frames**, re-origined so `u = 0` is the current
observation.

**Fitting (Algorithm 1).** `scipy.interpolate.generate_knots` inserts knots
adaptively until max reconstruction error `< max_error`, then `make_lsq_spline`
solves for control points. Per episode, whole episode, in frame-index time.

**Rollout (Algorithm 2).** The policy installs a segment at ~1-3 Hz; a daemon
thread in the env samples it at 100 Hz against the wall clock and calls
`servoJ`. The spline *is* the interpolator — no waypoint queue, no blending.

## Files

| Path | Role |
|---|---|
| `policy/bspline_policy/BSplineAction.py` | Algorithm 1, vendored from the reference |
| `policy/bspline_policy/BsplineAdapter.py` | RMB seam: loading, per-channel weights, time base, stats |
| `policy/bspline_policy/BsplinePolicyDataset.py` | one sample = (obs ending at `t`, params anchored at `t`) |
| `policy/bspline_policy/BsplineUnetPolicy.py` | DP U-Net + 1 knot channel, returns full horizon |
| `policy/bspline_policy/{Train,Rollout}BsplinePolicy.py` | training / Algorithm 2 |
| `envs/real/ur5e/RealUR5eBSplineEnvBase.py` | UR5e via `callm_controller`, 100 Hz sampler, dry run |
| `tests/TestBsplineAction.py` | 11 tests asserting the fitter's contract |

## Usage

```console
# train
$ python ./bin/Train.py BsplinePolicy --dataset_dir ./dataset/<name> --camera_names hand

# dry run: no hardware, prints what it would send
$ python ./bin/Rollout.py BsplinePolicy RealUR5eBSplineDemo \
    --config ./envs/configs/RealUR5eBSplineDemoEnv.yaml \
    --checkpoint <ckpt> --dry_run

# real robot
$ python ./bin/Rollout.py BsplinePolicy RealUR5eBSplineDemo \
    --config ./envs/configs/RealUR5eBSplineDemoEnv.yaml \
    --checkpoint <ckpt> --wait_before_start
```

Needs `robomimic==0.2.0 --no-deps` (DP's visual encoder), `scipy>=1.15`
(`generate_knots`), and `third_party/diffusion_policy`.

## Design choices

**`origin_time_scale` is measured, not assumed.** Knots are in frames, so
converting to wall clock needs the recording rate:
`u = elapsed_s × speedup × origin_time_scale`. It is measured from
`DataKey.TIME` at train time and stored in the checkpoint. Our UR5e records at
**30 Hz, not the 50 Hz** implied by `RealEnvBase.dt = 0.02` — the loop reads the
camera synchronously. A wrong value is silent: the robot traces the correct path
at the wrong speed.

**`--skip` does not stride the fit.** Knots are placed by trajectory curvature,
a property of the motion, not the sampling rate. Subsampling cuts samples
without cutting knots, so compression collapses. Observations still honour
`--skip`.

| | samples | knots | compression |
|---|---|---|---|
| full rate | 1045 | 193 | **5.41×** |
| `--skip 3` | 350 | 189 | 1.85× |

**Per-channel fit weights.** RMB packs arm + gripper into one vector: joints in
radians, gripper in Robotiq counts 0-255. One scalar `max_error` across those
units is dominated by the gripper. Channels are scaled by `w` before fitting and
control points divided by `w` after — exact, because a B-spline is linear in its
control points and the least-squares decouples per channel, so `w` only changes
*which knots get picked*. scipy has no per-channel weight (`make_lsq_spline(w=)`
is per-sample), so pre-scaling is the only way to express one.

Default `w_grip = 1e-3` → 2-count tolerance ≈ 0.39 mm (Hand-e) / 0.67 mm (2F-85).

**Action stats are per-channel, replicated across rows.** A knot vector is valid
only if non-decreasing, and a single positive affine map applied to every row
preserves that. Per-element stats (RMB's default) would give each row its own
scale and break monotonicity through normalization. `clip_sample: True` in the
DDIM config is likewise only sound because targets are in `[-1, 1]`.

**Action augmentation is off.** Gaussian noise on the knot column can break
monotonicity.

**`servoJ`, not `schedule_waypoint`.** `servoJ` → `drive_to_waypoint` is
latest-setpoint-wins, matching the reference's overwrite semantics. A waypoint
queue would double-buffer the plan and make the alignment re-origin ambiguous.

**`--max_plan_age` (new, not in the reference).** The reference replans only on
plan exhaustion. Our segments span ~1.2 s, so that would leave the policy
open-loop for over a second — segment length is an accident of how curvy the
demo was, not a design choice. `--max_plan_age 0.4` forces a replan on a timer
too. Safe because segment alignment already handles mid-plan installs.
*Note:* a span `S` is consumed in `S/m` at speedup `m`, so above ~3× exhaustion
always fires first and the timer goes inert.

## Findings and gotchas

**Chunk padding bug in the reference (fixed).** `chunk_bspline_trajectory`
right-padded short windows by repeating the final knot, producing knot
multiplicities up to `chunk_size + degree - 1`. A cubic B-spline is degenerate
past `degree + 1`, and the sliced control points no longer match the padded knot
vector. Measured on our data: **17.9 % of training targets degenerate, up to
161° of joint reconstruction error.** Fixed by not emitting windows that need
padding; the tail-reuse loop covers those timesteps with the last well-formed
chunk. After: 0 % degenerate, max error 0.101°.

**Two more reference bugs fixed.** `compress()` never forwarded `k=degree` to
scipy, so any non-cubic degree silently fitted a cubic. The tail loop relied on
loop variables leaking out of the chunk loop (`NameError` on a zero-chunk
episode).

**Knot projection moved into the core.** The paper's `u_i ← max(u_i, u_{i-1}+ε)`
lived only in the reference's rollout script; it is now
`project_knots_monotonic` in `BSplineAction.py` so training and rollout share it.

**Safety.** Driving the arm from the sampler thread bypasses
`RealEnvBase.overwrite_command_for_safety`. Replaced with: joint-limit box from
`action_space`, per-tick delta clamp (`max_joint_speed / spline_rate`), a
finiteness check, and a stale-plan watchdog that drops the segment if no new one
arrives within `stale_plan_timeout`. `JointInterpolationController` enforces
`max_joint_speed` independently.

The binding speed limit is **`max_joint_speed = 1.05 rad/s`, a conservative
`callm_controller` default — not UR5e hardware** (~π rad/s). Raise it in the
config if speedup saturates. The dry run reports what fraction of ticks hit the
clamp.

**Two pre-existing RMB limitations** hit on the way (not BSP-specific;
`DiffusionPolicyDataset` would hit both): `RmbData` honours `image_size` only for
the Compact `.rmb` format, not SingleHDF5 — so the dataset resizes itself; and
h5py fancy indexing rejects the duplicate indices that clamping an observation
window at an episode start produces — so unique indices are read and expanded.

**`RolloutBase`'s "Inference duration" is meaningless here.** Inference is async,
so that timer measures the submit. Real end-to-end latency is reported under
"B-spline rollout statistics".

## Measured baselines

5 grasping episodes, 30 Hz, `max_error=0.002`, `w_grip=1e-3`:

| | |
|---|---|
| compression | 4.16× |
| joint reconstruction | 0.101° max, 0.00022 rad RMS |
| gripper reconstruction | 1.93 counts (≈0.38 mm) |
| segment span | ~1.2 s at 1× |
| sampler loop (dry run) | 0.19 ms mean, 0 overruns at 100 Hz |

`tests/TestBsplineAction.py` asserts the fitter's **contract** — max error <
`max_error` per channel, in that channel's units (0.11459 deg for joints,
2.0 counts for the gripper) — not these values. Compression and where the error
lands inside the tolerance are both properties of the data: a curvier dataset
legitimately uses more of the budget. The table above is a reference point, not
a threshold. The suite also fails if any episode falls back to "best effort"
without reaching `max_error`, which is otherwise only a warning in the log.

## Known open points

- **Gripper is a slow ramp in our data** (~1.7 s, 5-count steps from spacemouse
  teleop), which is spline-friendly. Unweighted raw units cost only ~25 % of the
  compression, so the weight vector is an optimisation, not a rescue.
- **Hand-e → 2F-85 swap invalidates checkpoints.** Same count means a different
  opening on a 50 mm vs 85 mm stroke; the arm behaves, the grasp width does not.
- **`--speedup` untested on hardware.** Default 1.0. Re-check
  `span ≳ max_plan_age × speedup` when raising it.
- **A constant channel** (e.g. a gripper that never moved) is covered: the
  earlier no-gripper dataset passes the full suite, including knot monotonicity
  through normalization.
