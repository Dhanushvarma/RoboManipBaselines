# B-spline Policy (BSP)

Policy that predicts **continuous B-spline action curves** instead of discrete
action chunks, following [B-spline Policy: Accelerating Manipulation Policies via
B-spline Action Representations](https://B-spline-policy.github.io).

BSP is an action *representation*, not a network. This implementation wraps
RMB's Diffusion Policy backbone (`Diff.+BSP` in the paper): the network predicts
a fixed-size `(chunk_size + 2*degree, 1 + action_dim)` parameter matrix whose
column 0 is a knot vector and whose remaining columns are control points. With
the defaults (`chunk_size=10`, `degree=3`) and a 7-DoF UR5e action that is
`(16, 8)`.

The payoff is that a fitted curve can be sampled at any rate and retimed at
execution, so the controller receives smooth high-frequency commands even though
the policy runs slowly.

## Install

```console
$ pip install -e .[diffusion-policy]
$ pip install robomimic==0.2.0 --no-deps
$ git submodule update --init third_party/diffusion_policy
```

`robomimic` is needed for the Diffusion Policy visual observation encoder
(ResNet18 + `CropRandomizer`); `--no-deps` avoids its stale pins fighting the
versions RMB requires. `scipy >= 1.15` is also required, for
`scipy.interpolate.generate_knots`.

## Dataset

Model training
```console
$ python ./bin/Train.py BsplinePolicy \
  --dataset_dir ./dataset/<dataset_name> \
  --checkpoint_dir ./checkpoint/BsplinePolicy/<checkpoint_name> \
  --camera_names <camera_name>
```

## Key options

| Option | Default | Meaning |
|---|---|---|
| `--chunk_size` | 10 | interior knots per segment; `horizon = chunk_size + 2*bspline_degree` |
| `--bspline_degree` | 3 | cubic |
| `--max_error` | 0.002 | fitting tolerance in weighted action units (0.002 rad = 0.115 deg for joints) |
| `--gripper_weight` | 1e-3 | per-channel fit weight for gripper channels |
| `--gripper_action_idxes` | `[6]` | which action channels are grippers (UR5e single-arm layout) |
| `--bspline_stride` | 1 | knot-index step between consecutive segments |
| `--n_obs_steps` | 2 | observation history length |
| `--skip` | 3 | strides **observations only**; the spline is always fitted at full rate |

### Why the gripper needs a weight

RMB packs arm and gripper into one `command_joint_pos` vector: joints in
radians, gripper in Robotiq counts 0-255. A single scalar `max_error` across
those units is dominated by the gripper, so each channel is scaled by `w` before
fitting and the control points are divided by `w` afterwards. Because a B-spline
is linear in its control points and the least-squares decouples per channel this
is exact -- the only thing `w` changes is which knots the adaptive fitter picks.
scipy has no per-channel weight (`make_lsq_spline(w=...)` is per-sample), so
pre-scaling is the only way to express one.

The default `1e-3` with `max_error=0.002` gives a 2-count tolerance: ~0.39 mm on
a Hand-e (50 mm stroke), ~0.67 mm on a 2F-85 (85 mm).

### Why `--skip` does not stride the fit

The adaptive fitter places knots by trajectory curvature, which is a property of
the motion rather than the sampling rate. Subsampling therefore barely reduces
the knot count while cutting the sample count proportionally, and the
compression benefit evaporates. Measured on a 30 Hz UR5e dataset at
`max_error=0.002`:

| | samples | knots | compression |
|---|---|---|---|
| full rate | 1045 | 193 | 5.41x |
| `--skip 3` | 350 | 189 | 1.85x |

## Time base

Knots are in **dataset frames**, so converting them to wall-clock at rollout
needs `origin_time_scale`, the recording rate in Hz. It is **measured** from
`DataKey.TIME` at training time and stored in `model_meta_info`, never assumed:
`RealEnvBase.dt` nominally implies 50 Hz but the loop reads the camera
synchronously and achieves ~30 Hz. A wrong value is a silent time-scaling error
-- the robot traces the correct path at the wrong speed.

Training also reports the timing jitter and warns above 10%, because the fit
assumes a uniform dt.

## Tests

```console
$ python -m unittest robo_manip_baselines.tests.TestBsplineAction
```

Regression thresholds come from a real UR5e grasping dataset; point them
elsewhere with `RMB_BSPLINE_TEST_DATASET=/path/to/dataset`.

## Deviations from the reference implementation

The spline math is a faithful port, but four things changed:

1. `compress()` now forwards `k=degree` to scipy. The original did not, so any
   `degree != 3` silently fitted a cubic while the surrounding index arithmetic
   used `degree`.
2. Chunking no longer emits windows that need knot padding. The original
   right-padded by repeating the final knot, producing multiplicities up to
   `chunk_size + degree - 1`; a cubic is degenerate past `degree + 1`. Measured
   on the UR5e data this affected **17.9% of targets, with up to 161 deg** of
   reconstruction error.
3. The tail loop no longer depends on loop variables leaking out of the chunk
   loop, which raised `NameError` on an episode that produced no chunks.
4. Knot-validity projection (`project_knots_monotonic`) lives in the core rather
   than only in the rollout script, so training-time checks and rollout share
   one implementation.
