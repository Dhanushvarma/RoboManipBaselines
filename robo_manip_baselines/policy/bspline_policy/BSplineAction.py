"""B-spline action representation (Algorithm 1 of the B-spline Policy paper).

Vendored from the reference implementation
(``bspline_policy/common/bspline_action.py``) with the spline math kept
verbatim. The RMB-specific adaptation lives in :mod:`BsplineAdapter`, not here.

The policy-facing representation is a dense parameter matrix::

    (chunk_size + 2 * degree, 1 + action_dim)

Column 0 stores the knot vector, in units of **dataset frames** and re-origined
so that ``u = 0`` is the current observation. The remaining columns store
B-spline control points. With ``chunk_size=10, degree=3`` and a 7-DoF UR5e
action this is ``(16, 8)``.

Deviations from the reference implementation, all deliberate:

* The module-scope ``ReplayBuffer`` import is dropped (it was only a type hint).
* ``compress()`` now forwards ``k=self.degree`` to scipy. The original did not,
  so any ``degree != 3`` silently fitted a cubic while the surrounding index
  arithmetic used ``self.degree``.
* ``BSplineChunkSampler`` is split: the fitting/chunking half survives as
  :class:`BSplineChunkFitter`, which takes a list of per-episode arrays instead
  of a flat replay buffer. Observation windowing is dropped -- RMB's
  ``DatasetBase`` already does it.
* The tail loop of ``_preprocess_chunks`` no longer relies on loop variables
  leaking out of the ``for chunk in chunks`` block, which raised ``NameError``
  on an episode that produced no chunks.
* :func:`eval_bspline_at` and :func:`project_knots_monotonic` are new; see their
  docstrings.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import numpy as np
import torch
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline

from .BSplineKnots import decode_relative_knots, encode_relative_knots


class ScipyBSplineCompression:
    """Fit a multi-dimensional trajectory with a reduced-knot B-spline."""

    def __init__(self, degree: int = 3):
        self.degree = int(degree)
        self.spline = None
        self.knots = None
        # False when the adaptive loop exhausted its knot vectors without
        # reaching max_error and fell back to the best it found. That means the
        # tolerance is too tight for the data, and every guarantee downstream
        # (including the reconstruction test) is void for that episode.
        self.hit_tolerance = False

    def compress(
        self,
        data: np.ndarray,
        max_error: float = 0.01,
        verbose: bool = False,
        s: float = 1e-12,
    ) -> np.ndarray:
        """Adaptive knot insertion until max reconstruction error < max_error.

        ``data`` is ``(T, D)`` for one whole episode. The time base is the
        **sample index** (``np.arange(T)``), so knot units are dataset frames
        and a fixed, uniform dt is assumed -- see ``BsplineAdapter`` for the
        jitter check that validates that assumption.
        """
        t = np.arange(len(data))
        last_knots = None
        last_error = None
        self.hit_tolerance = False
        for knots in generate_knots(t, data, k=self.degree, s=s):
            spl = make_lsq_spline(t, data, knots, k=self.degree)
            pred_data = spl(t)
            error = np.abs(pred_data - data).max()
            last_knots = knots
            last_error = error
            if error < max_error:
                self.knots = knots
                self.spline = spl
                self.hit_tolerance = True
                break

        if self.knots is None:
            print(
                "Failing to compress trajectory with max error "
                f"{max_error}, use min error we can find. Error is {last_error}. "
                "You can try to increase the s value."
            )
            self.knots = last_knots
            self.spline = make_lsq_spline(t, data, self.knots, k=self.degree)

        if verbose:
            print(f"compression ratio: {len(self.knots) / len(t)}")

        return self.knots


def extract_unique_knots(t_full: np.ndarray, degree: int) -> np.ndarray:
    """Extract the unique knot span from FITPACK's repeated-boundary format."""
    return t_full[degree:-degree]


def whole_episode_params(compressor: ScipyBSplineCompression) -> np.ndarray:
    """Pack a whole-episode fit into one ``(n_knots, 1 + action_dim)`` matrix.

    Same layout as a training chunk -- column 0 knots, the rest control points --
    just longer, so it can be handed to anything that consumes a segment
    (``eval_bspline_at``, ``bspline_span``, the env's
    ``install_bspline_segment``). None of those hardcode the chunk length.

    scipy gives ``len(c) == len(t) - degree - 1``; the trailing rows are padded
    with the final control point exactly as ``chunk_bspline_trajectory`` does,
    and are discarded again on decode.

    Used by the replay diagnostic to play a whole episode as a single segment.
    """
    if compressor.spline is None:
        raise ValueError("Please call compress() before packing parameters")

    degree = compressor.degree
    t_full, c_full, _ = compressor.spline.tck

    params = np.zeros((len(t_full), 1 + c_full.shape[1]), dtype=np.float64)
    params[:, 0] = t_full
    params[: len(c_full), 1:] = c_full
    if len(c_full) < len(t_full):
        params[len(c_full) :, 1:] = c_full[-1]
    return params


def chunk_bspline_trajectory(
    compressor: ScipyBSplineCompression,
    chunk_size: int = 8,
    stride: Optional[int] = None,
    verbose: bool = False,
) -> list[dict]:
    """Split a fitted B-spline into fixed-size parameter chunks."""
    if compressor.spline is None:
        raise ValueError("Please call compress() before chunking")

    if stride is None:
        stride = chunk_size - 1

    degree = compressor.degree
    t_full, c_full, _ = compressor.spline.tck
    unique_t = extract_unique_knots(t_full, degree)
    n_unique = len(unique_t)
    chunks = []

    if verbose:
        print(
            f"B-spline chunking: len(t)={len(t_full)}, len(c)={len(c_full)}, "
            f"degree={degree}, unique_knots={n_unique}, chunk_size={chunk_size}, "
            f"stride={stride}"
        )

    expected_len = chunk_size + 2 * degree

    # The reference implementation loops to ``n_unique - 1`` and right-pads any
    # short window by repeating the final knot / control point. Near the end of
    # an episode that is unsound in two ways: it produces knot multiplicities up
    # to ``chunk_size + degree - 1`` (a cubic is degenerate past ``degree + 1``),
    # and it pads *past* FITPACK's clamped end-boundary knots, where the sliced
    # control points have already run out. Measured on the UR5e grasping
    # dataset, 17.9% of targets came out degenerate with up to 161 deg of joint
    # reconstruction error.
    #
    # A full window needs 16 knots and 12 control points from index ``j``; since
    # ``len(c_full) == len(t_full) - degree - 1`` both give the same bound. So
    # stop at the last index where no padding is required. The remaining
    # timesteps are covered by the tail-reuse loop in BSplineChunkFitter, which
    # reuses the last well-formed chunk.
    last_start = len(t_full) - expected_len
    if last_start < 0:
        raise ValueError(
            f"Episode fitted to only {len(t_full)} knots, fewer than the "
            f"{expected_len} a single chunk needs. Lower chunk_size or "
            f"max_error, or drop the episode."
        )

    for start_idx in range(0, min(n_unique - 1, last_start + 1), stride):
        chunk_t = t_full[start_idx : start_idx + expected_len]
        chunk_c = c_full[start_idx : start_idx + expected_len]

        if len(chunk_t) != expected_len:
            raise AssertionError("chunk_t length should equal chunk_size + 2 * degree")
        if len(chunk_c) < expected_len - degree - 1:
            raise AssertionError("not enough control points for this chunk")
        if len(chunk_c) < expected_len:
            # Only the trailing rows that decode_bspline_action discards are
            # missing; pad them so the matrix is rectangular.
            pad = np.repeat(chunk_c[-1:], expected_len - len(chunk_c), axis=0)
            chunk_c = np.concatenate([chunk_c, pad], axis=0)

        chunks.append({"t": chunk_t, "c": chunk_c, "k": degree})

    return chunks


def make_bspline_cache_path(
    base_path: str,
    filenames: list,
    chunk_size: int,
    degree: int,
    max_error: float,
    stride: int,
    weights: np.ndarray,
    relative_knots: bool = False,
) -> str:
    """Content-addressed cache path.

    Hashes the dataset file list and every parameter that changes the fit, so
    editing any of them invalidates the cache automatically. This is the RMB
    analogue of the reference implementation's
    ``make_bspline_sampler_cache_path``, which hashed a replay-buffer episode
    mask instead of filenames.
    """
    hasher = hashlib.sha1()
    # Hash the file list **in order**, not sorted. The fitted output is indexed
    # by episode position, and ``TrainBase.setup_rmb_files`` shuffles the file
    # list on every run -- so an order-insensitive key would let run 2 load a
    # fit built from run 1's ordering. The dataset would then pair observations
    # from ``filenames[i]`` with an action target fitted from a *different*
    # episode: silent, catastrophic, and invisible in the loss curve.
    for filename in filenames:
        hasher.update(os.path.basename(str(filename)).encode("utf-8"))
        hasher.update(b"\x00")
    hasher.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    hasher.update(
        (
            f"chunk={chunk_size}|degree={degree}|err={max_error}|stride={stride}|"
            f"relative_knots={int(relative_knots)}"
        ).encode("utf-8")
    )
    digest = hasher.hexdigest()[:16]
    return os.path.join(base_path, f"bspline_chunks_{digest}.npz")


class BSplineChunkFitter:
    """Fit every episode and expand into per-timestep B-spline parameter targets.

    This is Algorithm 1. Input is a list of ``(T_i, D)`` float arrays, one per
    episode, **already scaled by the per-channel weight vector** (see
    ``BsplineAdapter``). Output is a flat bank of
    ``(chunk_size + 2*degree, 1 + D)`` targets plus a timestep index.
    """

    def __init__(
        self,
        episode_actions_list: list,
        chunk_size: int = 10,
        degree: int = 3,
        max_error: float = 0.002,
        stride: int = 1,
        relative_knots: bool = False,
        cache_path: Optional[str] = None,
        verbose: bool = True,
    ):
        self.chunk_size = int(chunk_size)
        self.degree = int(degree)
        self.max_error = float(max_error)
        self.stride = int(stride)
        self.relative_knots = bool(relative_knots)
        self.verbose = bool(verbose)

        self.n_action_steps = self.chunk_size + 2 * self.degree

        if cache_path is not None and os.path.exists(cache_path):
            # episode_actions_list may be None here: a caller that already knows
            # the cache is warm can skip loading the episodes entirely.
            self._load_cache(cache_path)
            self.n_action_channels = int(self.all_actions.shape[2])
            return

        if episode_actions_list is None:
            raise ValueError(
                "[BSplineChunkFitter] episode_actions_list is required when "
                "there is no cache to load."
            )
        self.n_action_channels = 1 + int(episode_actions_list[0].shape[1])

        self._preprocess_chunks(episode_actions_list)

        if cache_path is not None:
            self._save_cache(cache_path)

    # ------------------------------------------------------------- caching ---

    def _load_cache(self, cache_path: str) -> None:
        if self.verbose:
            print(f"[BSplineChunkFitter] Loading cached chunks from {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            self.all_actions = data["all_actions"]
            self.timestep_to_chunk = data["timestep_to_chunk"]
            self.valid_timesteps = data["valid_timesteps"]
            self.episode_starts = data["episode_starts"]
            self.episode_lengths = data["episode_lengths"]
            self.n_knots_total = int(data["n_knots_total"])
            self.n_samples_total = int(data["n_samples_total"])
            self.fallback_episodes = data["fallback_episodes"].tolist()
        if self.verbose:
            print(
                f"[BSplineChunkFitter] {len(self.all_actions)} chunks, "
                f"{len(self.valid_timesteps)} valid timesteps"
            )

    def _save_cache(self, cache_path: str) -> None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp.npz"
        np.savez(
            tmp_path,
            all_actions=self.all_actions,
            timestep_to_chunk=self.timestep_to_chunk,
            valid_timesteps=self.valid_timesteps,
            episode_starts=np.asarray(self.episode_starts, dtype=np.int64),
            episode_lengths=np.asarray(self.episode_lengths, dtype=np.int64),
            n_knots_total=np.int64(self.n_knots_total),
            n_samples_total=np.int64(self.n_samples_total),
            fallback_episodes=np.asarray(self.fallback_episodes, dtype=np.int64),
        )
        os.replace(tmp_path, cache_path)
        if self.verbose:
            print(f"[BSplineChunkFitter] Saved cached chunks to {cache_path}")

    # ------------------------------------------------------------ the fit ----

    def _preprocess_chunks(self, episode_actions_list: list) -> None:
        if self.verbose:
            print(
                f"[BSplineChunkFitter] Fitting {len(episode_actions_list)} episodes "
                f"(chunk_size={self.chunk_size}, degree={self.degree}, "
                f"max_error={self.max_error}, stride={self.stride})"
            )

        episode_lengths = [len(a) for a in episode_actions_list]
        episode_starts = np.concatenate([[0], np.cumsum(episode_lengths)[:-1]]).astype(
            np.int64
        )
        total_steps = int(np.sum(episode_lengths))

        all_chunks = []
        self.timestep_to_chunk = np.full(total_steps, -1, dtype=np.int64)
        n_knots_total = 0
        self.fallback_episodes = []

        for ep_idx, episode_actions in enumerate(episode_actions_list):
            ep_start = int(episode_starts[ep_idx])
            ep_length = len(episode_actions)

            compressor = ScipyBSplineCompression(degree=self.degree)
            compressor.compress(
                episode_actions, max_error=self.max_error, verbose=False
            )
            n_knots_total += len(compressor.knots)
            if not compressor.hit_tolerance:
                self.fallback_episodes.append(ep_idx)

            chunks = chunk_bspline_trajectory(
                compressor,
                chunk_size=self.chunk_size,
                stride=self.stride,
                verbose=False,
            )
            if len(chunks) == 0:
                raise ValueError(
                    f"[BSplineChunkFitter] Episode {ep_idx} produced no chunks "
                    f"(length {ep_length}); it is too short to fit."
                )

            # Initialised before the loop so the tail loop below cannot depend
            # on a leaked loop variable (the reference implementation did).
            chunk_data = None
            local_idx_in_episode = 0

            for chunk in chunks:
                chunk_data = np.zeros(
                    (self.n_action_steps, self.n_action_channels), dtype=np.float32
                )
                t_timesteps = chunk["t"]
                chunk_data[:, 0] = t_timesteps.copy()
                chunk_data[:, 1:] = chunk["c"]

                while local_idx_in_episode <= t_timesteps[self.degree]:
                    if local_idx_in_episode >= ep_length:
                        break
                    all_chunks.append(
                        self._localise(chunk_data, local_idx_in_episode)
                    )
                    self.timestep_to_chunk[local_idx_in_episode + ep_start] = (
                        len(all_chunks) - 1
                    )
                    local_idx_in_episode += 1

            # Tail: reuse the last chunk for the remaining timesteps. Its knots
            # go negative, which is valid -- the plan simply started in the past.
            while local_idx_in_episode < ep_length:
                all_chunks.append(self._localise(chunk_data, local_idx_in_episode))
                self.timestep_to_chunk[local_idx_in_episode + ep_start] = (
                    len(all_chunks) - 1
                )
                local_idx_in_episode += 1

        self.all_actions = np.asarray(all_chunks, dtype=np.float32)
        self.episode_starts = episode_starts
        self.episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
        self.valid_timesteps = np.flatnonzero(self.timestep_to_chunk >= 0)
        self.n_knots_total = n_knots_total
        self.n_samples_total = total_steps

        if self.fallback_episodes:
            print(
                f"[BSplineChunkFitter] WARNING: {len(self.fallback_episodes)} of "
                f"{len(episode_actions_list)} episodes did not reach max_error="
                f"{self.max_error} (indexes {self.fallback_episodes}). Their "
                f"targets exceed the tolerance; raise max_error or check the data."
            )

        if self.verbose:
            print(
                f"[BSplineChunkFitter] {len(self.all_actions)} chunks, "
                f"shape {self.all_actions.shape}, "
                f"compression {self.compression_ratio:.2f}x"
            )

    def _localise(self, chunk_data: np.ndarray, local_idx: int) -> np.ndarray:
        """Re-origin a chunk's knot column so u=0 is the given timestep."""
        local_chunk_data = chunk_data.copy()
        local_chunk_data[:, 0] -= local_idx
        if self.relative_knots:
            local_chunk_data = encode_relative_knots(
                local_chunk_data, degree=self.degree
            )
        return local_chunk_data

    # ------------------------------------------------------------ accessors --

    @property
    def compression_ratio(self) -> float:
        """Original samples divided by knot-control-point pairs."""
        return self.n_samples_total / max(self.n_knots_total, 1)

    def get_chunk(self, episode_idx: int, time_idx: int) -> np.ndarray:
        """B-spline parameter target for one (episode, timestep)."""
        flat_idx = int(self.episode_starts[episode_idx]) + int(time_idx)
        chunk_idx = self.timestep_to_chunk[flat_idx]
        if chunk_idx < 0:
            raise IndexError(
                f"No chunk for episode {episode_idx} timestep {time_idx}"
            )
        return self.all_actions[chunk_idx]

    def get_action_stats(self) -> dict:
        """Per-element min/max/mean/std over the parameter bank, keepdims."""
        if len(self.all_actions) == 0:
            shape = (1, self.n_action_steps, self.n_action_channels)
            return {
                "min": np.zeros(shape, dtype=np.float32),
                "max": np.ones(shape, dtype=np.float32),
                "mean": np.zeros(shape, dtype=np.float32),
                "std": np.ones(shape, dtype=np.float32),
            }

        return {
            "min": np.min(self.all_actions, axis=0, keepdims=True),
            "max": np.max(self.all_actions, axis=0, keepdims=True),
            "mean": np.mean(self.all_actions, axis=0, keepdims=True),
            "std": np.std(self.all_actions, axis=0, keepdims=True),
        }


def project_knots_monotonic(action_params, degree: int = 3, eps: float = 1e-6):
    """Enforce a non-decreasing knot vector: ``u_i <- max(u_i, u_{i-1} + eps)``.

    This is the knot-validity projection of the paper (App. A.1). The reference
    implementation has it as ``safer_knots`` inside the rollout script rather
    than in the core, so training-time validation and rollout could drift apart;
    it lives here so both share one implementation.

    Operates on the knot column of a ``(L, 1 + action_dim)`` matrix in place on a
    copy, and returns the copy.
    """
    is_tensor = torch.is_tensor(action_params)
    params = (
        action_params.detach().cpu().numpy().copy()
        if is_tensor
        else np.array(action_params, dtype=np.float64, copy=True)
    )

    knots = params[..., 0]
    for idx in range(1, knots.shape[-1]):
        knots[..., idx] = np.maximum(knots[..., idx], knots[..., idx - 1] + eps)

    if is_tensor:
        return torch.as_tensor(
            params, dtype=action_params.dtype, device=action_params.device
        )
    return params


def _to_bspline(action_params, degree: int, relative_knots: bool):
    """Build a scipy BSpline from one parameter matrix."""
    if torch.is_tensor(action_params):
        action_params = action_params.detach().cpu().numpy()
    action_params = np.asarray(action_params, dtype=np.float64)
    if relative_knots:
        action_params = decode_relative_knots(action_params, degree=degree)

    knots = action_params[:, 0].copy()
    control_points = action_params[: -(degree + 1), 1:].copy()
    t_min = knots[degree]
    t_max = knots[-(degree + 1)]
    return BSpline(knots, control_points, degree, extrapolate=False), t_min, t_max


def decode_bspline_action(
    action_params,
    degree: int = 3,
    num_actions: int = 8,
    relative_knots: bool = False,
) -> np.ndarray:
    """Decode one B-spline parameter matrix into regular action vectors."""
    spline, t_min, t_max = _to_bspline(action_params, degree, relative_knots)
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")

    if num_actions <= 1:
        t_eval = np.asarray([t_min], dtype=np.float64)
    else:
        t_eval = np.linspace(t_min, t_max, int(num_actions), dtype=np.float64)
    return spline(t_eval).astype(np.float32)


def eval_bspline_at(
    action_params,
    u,
    degree: int = 3,
    relative_knots: bool = False,
    clamp: bool = True,
) -> np.ndarray:
    """Evaluate a B-spline segment at arbitrary parameter value(s) ``u``.

    ``decode_bspline_action`` only supports uniform sampling across the whole
    valid span, which is not enough for Algorithm 2: the 100 Hz sampler needs an
    arbitrary wall-clock-derived ``u``, and the segment-alignment search needs to
    probe candidate ``u`` values. With ``clamp=True`` (the default) ``u`` is
    clipped into ``[t_min, t_max]`` so evaluation never returns NaN from
    ``extrapolate=False``.

    Returns ``(action_dim,)`` for scalar ``u``, else ``(len(u), action_dim)``.
    """
    spline, t_min, t_max = _to_bspline(action_params, degree, relative_knots)
    if t_max <= t_min:
        raise ValueError(f"Invalid B-spline range: [{t_min}, {t_max}]")

    u_arr = np.atleast_1d(np.asarray(u, dtype=np.float64))
    if clamp:
        u_arr = np.clip(u_arr, t_min, t_max)

    values = spline(u_arr)
    if np.isscalar(u) or np.ndim(u) == 0:
        return values[0]
    return values


def bspline_span(action_params, degree: int = 3, relative_knots: bool = False):
    """Return ``(t_min, t_max)``, the valid parameter span of a segment."""
    _, t_min, t_max = _to_bspline(action_params, degree, relative_knots)
    return float(t_min), float(t_max)
