"""Tests for the B-spline action representation (Algorithm 1).

Structural tests run anywhere. The regression tests need an RMB-format UR5e
dataset and are skipped without one; point them at a dataset with::

    RMB_BSPLINE_TEST_DATASET=/path/to/dataset python -m unittest \
        robo_manip_baselines.tests.TestBsplineAction

The regression thresholds come from measurements on
``sample_dataset/RealUR5eDemo_20260728_162024_grippermoves`` (5 episodes,
29.98 Hz, gripper 0-255) and are documented in
``docs/bsp_rmb_integration_plan.md`` section 0.7.
"""

import os
import unittest

import numpy as np

from robo_manip_baselines.common import (
    DataKey,
    denormalize_data,
    find_rmb_files,
    normalize_data,
)
from robo_manip_baselines.policy.bspline_policy import BsplineAdapter as adapter
from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
    BSplineChunkFitter,
    bspline_span,
    decode_bspline_action,
    eval_bspline_at,
    project_knots_monotonic,
)

DATASET_ENV_VAR = "RMB_BSPLINE_TEST_DATASET"
DEFAULT_DATASET = os.path.join(
    os.path.dirname(__file__),
    "../../../sample_dataset/RealUR5eDemo_20260728_162024_grippermoves",
)

CHUNK_SIZE = 10
DEGREE = 3
MAX_ERROR = 0.002
STRIDE = 1

# These assert the fitter's *contract*, which is dataset-independent:
#   max|reconstruction error| < max_error, per channel, in that channel's units.
#
# An earlier version asserted measured values from one dataset (4.16x
# compression, 0.101 deg) which made these dataset fingerprints rather than
# regression tests -- a curvier dataset legitimately lands anywhere inside the
# tolerance, and its compression ratio is purely a property of the trajectories.
JOINT_TOLERANCE_DEG = np.rad2deg(MAX_ERROR)  # 0.11459 deg
GRIPPER_TOLERANCE_CNT = MAX_ERROR / 1.0e-3  # 2.0 counts at the default weight

# Reference values on sample_dataset/RealUR5eDemo_20260728_162024_grippermoves,
# recorded for comparison only -- not asserted.
REFERENCE_COMPRESSION = 4.16
REFERENCE_JOINT_MAX_DEG = 0.101


def _report(message):
    """One short status line per test, so a run is readable at a glance."""
    print(f"\n    -> {message}", flush=True)


def _dataset_dir():
    path = os.environ.get(DATASET_ENV_VAR, DEFAULT_DATASET)
    path = os.path.abspath(path)
    return path if os.path.isdir(path) else None


class TestBsplineStructure(unittest.TestCase):
    """Tests that need no dataset."""

    @staticmethod
    def _synthetic_episode(length=200, action_dim=7):
        t = np.arange(length)
        joints = np.stack([np.sin(0.05 * t + p) * 1.2 for p in range(action_dim - 1)], 1)
        gripper = np.clip((t - length // 2) / 10.0, 0, 1)[:, None] * 255.0
        return np.concatenate([joints, gripper], axis=1)

    def test_target_shape_and_full_coverage(self):
        weights = adapter.build_action_weights(
            7, adapter.DEFAULT_GRIPPER_ACTION_IDXES, adapter.DEFAULT_GRIPPER_WEIGHT
        )
        episodes = [self._synthetic_episode() * weights for _ in range(2)]
        fitter = BSplineChunkFitter(
            episodes, CHUNK_SIZE, DEGREE, MAX_ERROR, STRIDE, verbose=False
        )

        self.assertEqual(
            fitter.all_actions.shape[1:], (CHUNK_SIZE + 2 * DEGREE, 8)
        )
        # Every timestep of every episode must map to a chunk.
        self.assertEqual(len(fitter.valid_timesteps), fitter.n_samples_total)
        self.assertTrue(np.all(fitter.timestep_to_chunk >= 0))
        _report(
            f"target {fitter.all_actions.shape[1:]}, "
            f"{fitter.n_samples_total} timesteps all mapped"
        )

    def test_no_degenerate_knot_vectors(self):
        """No chunk may exceed knot multiplicity degree+1.

        The reference implementation right-pads short windows by repeating the
        final knot, which produces multiplicities up to chunk_size+degree-1. A
        cubic B-spline is degenerate past degree+1 and evaluation blows up.
        """
        weights = adapter.build_action_weights(
            7, adapter.DEFAULT_GRIPPER_ACTION_IDXES, adapter.DEFAULT_GRIPPER_WEIGHT
        )
        episodes = [self._synthetic_episode(n) * weights for n in (150, 200, 260)]
        fitter = BSplineChunkFitter(
            episodes, CHUNK_SIZE, DEGREE, MAX_ERROR, STRIDE, verbose=False
        )

        knots = fitter.all_actions[:, :, 0]
        worst = 0
        for row in knots:
            _, counts = np.unique(np.round(row, 9), return_counts=True)
            worst = max(worst, int(counts.max()))
            self.assertLessEqual(
                int(counts.max()),
                DEGREE + 1,
                msg=f"knot multiplicity {counts.max()} exceeds degree+1 in {row}",
            )
        _report(
            f"max knot multiplicity {worst} over {len(knots)} chunks "
            f"(limit {DEGREE + 1})"
        )

    def test_knots_are_non_decreasing(self):
        weights = adapter.build_action_weights(
            7, adapter.DEFAULT_GRIPPER_ACTION_IDXES, adapter.DEFAULT_GRIPPER_WEIGHT
        )
        episodes = [self._synthetic_episode() * weights]
        fitter = BSplineChunkFitter(
            episodes, CHUNK_SIZE, DEGREE, MAX_ERROR, STRIDE, verbose=False
        )
        diffs = np.diff(fitter.all_actions[:, :, 0], axis=1)
        self.assertGreaterEqual(float(diffs.min()), 0.0)
        n_repeat = int((diffs == 0.0).sum())
        _report(
            f"min knot spacing {diffs.min():+.4f} over {len(diffs)} chunks "
            f"({n_repeat} repeats, all at clamped boundaries)"
        )

    def test_project_knots_monotonic(self):
        params = np.zeros((16, 8), dtype=np.float64)
        params[:, 0] = np.arange(16)[::-1]  # strictly decreasing -> invalid
        fixed = project_knots_monotonic(params, degree=DEGREE)
        self.assertTrue(np.all(np.diff(fixed[:, 0]) > 0))
        # An already-valid vector must be left alone.
        good = np.zeros((16, 8))
        good[:, 0] = np.arange(16)
        np.testing.assert_allclose(
            project_knots_monotonic(good, degree=DEGREE)[:, 0], good[:, 0]
        )
        _report("reversed knots repaired; valid knots left untouched")

    def test_eval_matches_decode(self):
        """eval_bspline_at on a uniform grid must equal decode_bspline_action."""
        weights = adapter.build_action_weights(
            7, adapter.DEFAULT_GRIPPER_ACTION_IDXES, adapter.DEFAULT_GRIPPER_WEIGHT
        )
        fitter = BSplineChunkFitter(
            [self._synthetic_episode() * weights],
            CHUNK_SIZE,
            DEGREE,
            MAX_ERROR,
            STRIDE,
            verbose=False,
        )
        params = fitter.get_chunk(0, 40)
        t_min, t_max = bspline_span(params, degree=DEGREE)
        expected = decode_bspline_action(params, degree=DEGREE, num_actions=8)
        actual = eval_bspline_at(
            params, np.linspace(t_min, t_max, 8), degree=DEGREE
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        _report(
            f"eval_bspline_at == decode_bspline_action over span "
            f"[{t_min:.1f}, {t_max:.1f}], max diff "
            f"{np.abs(actual - expected).max():.2e}"
        )

    def test_per_channel_weighting_is_exact(self):
        """Unscaling control points must recover the unweighted curve exactly."""
        from scipy.interpolate import BSpline, make_lsq_spline

        from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
            ScipyBSplineCompression,
        )

        data = self._synthetic_episode(180)
        weights = adapter.build_action_weights(
            7, adapter.DEFAULT_GRIPPER_ACTION_IDXES, adapter.DEFAULT_GRIPPER_WEIGHT
        )
        t = np.arange(len(data))

        comp = ScipyBSplineCompression(degree=DEGREE)
        comp.compress(data * weights, max_error=MAX_ERROR)
        unscaled = comp.spline.c / weights

        # Fitting the raw data on the same knots must give the same coefficients.
        direct = make_lsq_spline(t, data, comp.spline.t, k=DEGREE)
        np.testing.assert_allclose(unscaled, direct.c, rtol=0, atol=1e-9)

        curve = BSpline(comp.spline.t, unscaled, DEGREE)(t)
        self.assertLess(np.abs(curve[:, :6] - data[:, :6]).max(), MAX_ERROR * 1.05)
        _report(
            f"unscale(fit(scaled)) == fit(raw) to "
            f"{np.abs(unscaled - direct.c).max():.2e}"
        )


@unittest.skipIf(_dataset_dir() is None, "no RMB dataset available")
class TestBsplineRegression(unittest.TestCase):
    """Regression against measured baselines on the real UR5e dataset."""

    @classmethod
    def setUpClass(cls):
        cls.filenames = sorted(find_rmb_files(_dataset_dir()))
        cls.weights = adapter.build_action_weights(
            7, adapter.DEFAULT_GRIPPER_ACTION_IDXES, adapter.DEFAULT_GRIPPER_WEIGHT
        )
        cls.raw = adapter.load_episode_actions(
            cls.filenames, [DataKey.COMMAND_JOINT_POS], weights=None, verbose=False
        )
        weighted = [ep * cls.weights for ep in cls.raw]
        cls.fitter = BSplineChunkFitter(
            weighted, CHUNK_SIZE, DEGREE, MAX_ERROR, STRIDE, verbose=False
        )

    def test_time_base_is_measured_not_nominal(self):
        info = adapter.measure_time_base(self.filenames, verbose=False)
        # Recorded at ~30 Hz, NOT the nominal 50 Hz implied by RealEnvBase.dt.
        self.assertAlmostEqual(info["origin_time_scale"], 29.98, delta=0.5)
        self.assertLess(info["jitter_p95"], 0.05)
        _report(
            f"origin_time_scale {info['origin_time_scale']:.2f} Hz "
            f"(nominal RealEnvBase.dt implies 50), jitter p95 "
            f"{info['jitter_p95'] * 100:.1f}%"
        )

    def test_every_episode_reached_tolerance(self):
        """No episode may fall back to 'best effort'.

        ``compress()`` gives up if it exhausts its knot vectors before reaching
        ``max_error``, keeps the best fit found, and prints a warning that is
        easy to miss in training output. When that happens the reconstruction
        guarantee is void for that episode, so fail loudly instead.
        """
        self.assertEqual(
            self.fitter.fallback_episodes,
            [],
            msg=(
                f"episodes {self.fitter.fallback_episodes} did not reach "
                f"max_error={MAX_ERROR}; raise the tolerance or inspect the data"
            ),
        )
        _report(
            f"all {len(self.raw)} episodes reached max_error={MAX_ERROR} "
            f"(no best-effort fallbacks)"
        )

    def test_compression_ratio_is_a_compression(self):
        """The representation must be smaller than what it replaces.

        Deliberately not an equality: the ratio depends on how curvy the
        demonstrations are, so pinning it would make this a dataset fingerprint.
        A ratio at or below 1.0 means the fit bought nothing and max_error is
        far too tight for the data.
        """
        ratio = self.fitter.compression_ratio
        self.assertGreater(ratio, 1.0)
        _report(
            f"compression {ratio:.2f}x "
            f"({self.fitter.n_samples_total} samples -> "
            f"{self.fitter.n_knots_total} knots; reference dataset "
            f"{REFERENCE_COMPRESSION:.2f}x)"
        )

    def test_reconstruction_within_tolerance(self):
        """Chunked targets must reproduce the demos at exact frame offsets."""
        joint_err = gripper_err = 0.0
        for ep_idx, episode in enumerate(self.raw):
            n = len(episode)
            for time_idx in range(n):
                params = self.fitter.get_chunk(ep_idx, time_idx)
                t_min, t_max = bspline_span(params, degree=DEGREE)
                offsets = np.arange(
                    np.ceil(t_min), np.floor(min(t_max, n - 1 - time_idx)) + 1
                )
                if len(offsets) == 0:
                    continue
                pred = adapter.unweight_actions(
                    eval_bspline_at(params, offsets, degree=DEGREE), self.weights
                )
                ref = episode[(time_idx + offsets).astype(int)]
                joint_err = max(joint_err, np.abs(pred[:, :6] - ref[:, :6]).max())
                gripper_err = max(gripper_err, np.abs(pred[:, 6] - ref[:, 6]).max())

        joint_err_deg = np.rad2deg(joint_err)
        _report(
            f"joint {joint_err_deg:.5f}/{JOINT_TOLERANCE_DEG:.5f} deg, "
            f"gripper {gripper_err:.3f}/{GRIPPER_TOLERANCE_CNT:.3f} cnt "
            f"(used {100 * joint_err_deg / JOINT_TOLERANCE_DEG:.0f}% of the "
            f"joint budget)"
        )
        # Small slack for the difference between the fit's own error metric
        # (evaluated on the whole episode) and this per-chunk re-evaluation.
        self.assertLess(joint_err_deg, JOINT_TOLERANCE_DEG * 1.02)
        self.assertLess(gripper_err, GRIPPER_TOLERANCE_CNT * 1.02)

    def test_cache_is_order_sensitive_and_exact(self):
        """A cached fit must equal a fresh one, and must key on file *order*.

        The fitted output is indexed by episode position, and
        ``TrainBase.setup_rmb_files`` shuffles the file list on every run. An
        order-insensitive cache key would therefore let a later run load a fit
        built from an earlier run's ordering, pairing observations from one
        episode with an action target fitted from another -- silent, and
        invisible in the loss curve.
        """
        import tempfile

        from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
            make_bspline_cache_path,
        )

        bspline_info = {
            "chunk_size": CHUNK_SIZE,
            "degree": DEGREE,
            "max_error": MAX_ERROR,
            "stride": STRIDE,
            "relative_knots": False,
            "action_weights": self.weights.tolist(),
        }
        reversed_files = list(reversed(self.filenames))
        key = make_bspline_cache_path(
            "/tmp", self.filenames, CHUNK_SIZE, DEGREE, MAX_ERROR, STRIDE, self.weights
        )
        key_reversed = make_bspline_cache_path(
            "/tmp", reversed_files, CHUNK_SIZE, DEGREE, MAX_ERROR, STRIDE, self.weights
        )
        self.assertNotEqual(
            key, key_reversed, msg="cache key must depend on file order"
        )

        with tempfile.TemporaryDirectory() as cache_dir:
            cold = adapter.build_fitter(
                self.filenames,
                [DataKey.COMMAND_JOINT_POS],
                bspline_info,
                cache_dir=cache_dir,
                verbose=False,
            )
            warm = adapter.build_fitter(
                self.filenames,
                [DataKey.COMMAND_JOINT_POS],
                bspline_info,
                cache_dir=cache_dir,
                verbose=False,
            )

        np.testing.assert_array_equal(warm.all_actions, cold.all_actions)
        np.testing.assert_array_equal(
            warm.timestep_to_chunk, cold.timestep_to_chunk
        )
        np.testing.assert_array_equal(warm.episode_lengths, cold.episode_lengths)
        self.assertEqual(warm.fallback_episodes, cold.fallback_episodes)
        _report(
            f"cache round-trip exact over {len(warm.all_actions)} chunks; "
            f"reversing the file list changes the key"
        )

    def test_normalization_preserves_knot_monotonicity(self):
        """The whole point of per-channel (not per-element) action stats."""
        stats = adapter.build_action_stats(self.fitter, "limits")
        self.assertTrue(
            all(
                np.allclose(stats[key], stats[key][0])
                for key in ("min", "max", "mean", "std")
            ),
            msg="action stats must be per-channel replicated across rows",
        )

        # Spread across whatever this dataset's size happens to be.
        n_chunks = len(self.fitter.all_actions)
        for idx in np.linspace(0, n_chunks - 1, 8, dtype=int):
            params = self.fitter.all_actions[idx].astype(np.float64)
            self.assertTrue(np.all(np.diff(params[:, 0]) >= 0))
            norm = normalize_data(params, stats)
            self.assertTrue(
                np.all(np.diff(norm[:, 0]) >= 0),
                msg="normalization broke knot monotonicity",
            )
            self.assertLessEqual(np.abs(norm).max(), 1.0 + 1e-6)
            np.testing.assert_allclose(
                denormalize_data(norm, stats), params, rtol=0, atol=1e-9
            )
        _report(
            f"stats per-channel replicated; knots stayed monotonic through "
            f"normalize/denormalize on {len(np.linspace(0, n_chunks - 1, 8, dtype=int))} chunks"
        )


if __name__ == "__main__":
    unittest.main()
