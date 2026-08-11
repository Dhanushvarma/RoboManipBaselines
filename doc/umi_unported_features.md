# UMI features not ported into `UmiDp`

These exist in UMI but are **disabled in its own released config**, so skipping them costs no fidelity.
Listed for the record in case one becomes relevant.

Paths are relative to `third_party/universal_manipulation_interface/diffusion_policy/`.

| Feature | UMI's value | What it does | Where |
| --- | --- | --- | --- |
| `inpaint_fixed_action_prefix` | `False`, and `eval_real.py` never passes `fixed_action_prefix` | Pins the already-committed tail of the previous chunk as a diffusion constraint, for continuity across inferences | `policy/diffusion_unet_timm_policy.py:30` |
| `repeat_frame_prob` | `0.0` | Repeats observation frames before the first grasp, to augment long idle approach phases | `common/sampler.py:32` |
| `max_duration` | `None` | Truncates episodes to a fixed length | `common/sampler.py:33` |
| `temporally_independent_normalization` | `False` | Fits normalization per timestep instead of flattening `(B,T,D)` | `dataset/umi_dataset.py:36` |
| `ignore_by_policy` / `ignore_proprioception` | `False` | Drops proprioception from the policy input while still using it for the relative transform (vision-only training) | `config/task/umi.yaml:12,30` |
| `robot0_eef_pos_abs`, `robot0_eef_rot_axis_angle_abs` | commented out | Absolute pose components supplied alongside the relative ones | `config/task/umi.yaml:31,38` |

## Also not ported, by decision

| Feature | Reason |
| --- | --- |
| zarr `ReplayBuffer` + `SequenceSampler` | Replaced by `UmiDpDataset` reading `.rmb` — the point of the integration |
| Per-key `down_sample_steps` | All keys use `3` in `umi.yaml`; RMB's global `--skip 3` is identical in effect |
| Multiple cameras | Single-arm, single-camera scope; `umi.yaml` also has only `camera0_rgb` |
| Multi-robot / bimanual | Single-arm scope |
| Transformer policy variant | UMI's config uses the UNet variant |
| `ignore_rgb` during normalizer fitting | Speed optimization only; same resulting statistics |

## One deliberate divergence

`action_padding` (`common/sampler.py:31`) defaults to `False` in UMI, which **drops** chunks whose action
window runs past the episode end. `UmiDpDataset` clips indices instead, i.e. pads with the last action,
keeping those chunks.

With 53–74 step episodes at `skip=3` and `action_horizon=16`, dropping would discard roughly the last 16
steps of every episode. Clipping was chosen to retain them. Switching is a one-line change in
`UmiDpDataset.setup_variables`.

## Separate matter

Per-key latency alignment is *not* in this category — it is genuinely unimplemented and deferred.
See [umi_latency_alignment.md](./umi_latency_alignment.md).
