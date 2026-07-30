import argparse
import os

import matplotlib
import matplotlib.ticker
import numpy as np

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files
from robo_manip_baselines.policy.bspline_policy import BsplineAdapter as adapter
from robo_manip_baselines.policy.bspline_policy.BSplineAction import (
    BSplineChunkFitter,
    ScipyBSplineCompression,
    bspline_span,
    eval_bspline_at,
    whole_episode_params,
)

# Reference data-viz palette, slots 1-2 (documented all-pairs validated).
# Identity is carried by two series only; everything else is ink or surface.
COLOR_DEMO = "#2a78d6"
COLOR_FIT = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Plot B-spline fit quality on demonstration data. Offline: no robot, "
            "no policy, no checkpoint. Answers 'is the representation lossy', "
            "independently of how well a policy learned to predict it."
        ),
    )
    parser.add_argument(
        "path",
        type=str,
        help="path to data (*.hdf5 or *.rmb) or directory containing them",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bspline_fit_plots",
        help="directory to write PNGs into",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display windows instead of only saving (needs a display; avoid "
        "over ssh -X, where pushing figures is very slow)",
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        nargs="*",
        default=None,
        help="which episodes to plot in detail (default: the first 5). The "
        "dataset summary always uses every episode",
    )
    parser.add_argument(
        "--no_summary", action="store_true", help="skip the dataset summary figure"
    )

    # Fit parameters: defaults match training.
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--bspline_degree", type=int, default=3)
    parser.add_argument("--max_error", type=float, default=0.002)
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
    parser.add_argument(
        "--max_plan_age",
        type=float,
        default=0.4,
        help="drawn as a reference line on the segment-span plot",
    )
    return parser.parse_args()


class PlotBsplineFit:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        if not self.show:
            matplotlib.use("agg")

        self.filenames = find_rmb_files(self.path)
        self.weights = adapter.build_action_weights(
            7, self.gripper_action_idxes, self.gripper_weight
        )
        self.joint_tol_deg = np.rad2deg(self.max_error)
        self.gripper_tol_cnt = self.max_error / self.gripper_weight

    # -------------------------------------------------------------- data ----

    def fit_episode(self, filename):
        """Fit one episode and collect everything the figures need."""
        with RmbData(filename) as rmb_data:
            time_seq = np.asarray(rmb_data[DataKey.TIME][:], dtype=np.float64)
            demo = np.asarray(
                rmb_data[DataKey.COMMAND_JOINT_POS][:], dtype=np.float64
            )

        compressor = ScipyBSplineCompression(degree=self.bspline_degree)
        compressor.compress(demo * self.weights, max_error=self.max_error)

        params = whole_episode_params(compressor)
        frames = np.arange(len(demo))
        fitted = np.asarray(
            eval_bspline_at(params, frames, degree=self.bspline_degree)
        ) / self.weights

        # Interior knots only: the clamped boundary repeats are an artefact of
        # FITPACK's format, not places the fitter chose to spend capacity.
        knots = np.unique(compressor.knots[self.bspline_degree : -self.bspline_degree])

        fitter = BSplineChunkFitter(
            [demo * self.weights],
            chunk_size=self.chunk_size,
            degree=self.bspline_degree,
            max_error=self.max_error,
            stride=self.bspline_stride,
            verbose=False,
        )
        spans = []
        for time_idx in range(0, len(demo), 5):
            lo, hi = bspline_span(
                fitter.get_chunk(0, time_idx), degree=self.bspline_degree
            )
            if hi > lo:
                spans.append(hi - lo)

        rate = 1.0 / np.median(np.diff(time_seq))
        return {
            "name": os.path.splitext(os.path.basename(filename))[0],
            "time": time_seq - time_seq[0],
            "demo": demo,
            "fitted": fitted,
            "knots": knots,
            "n_knots": len(compressor.knots),
            "n_frames": len(demo),
            "compression": len(demo) / len(compressor.knots),
            "hit_tolerance": compressor.hit_tolerance,
            "spans_s": np.array(spans) / rate,
            "rate": rate,
        }

    # ------------------------------------------------------------- style ----

    @staticmethod
    def style_axis(ax, ylabel=None, xlabel=None, title=None):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8, length=3)
        if ylabel:
            ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
        if title:
            ax.set_title(title, color=INK, fontsize=10, loc="left")

    # ----------------------------------------------------- per-episode ------

    def plot_episode(self, data):
        import matplotlib.pylab as plt

        fig = plt.figure(figsize=(15.0, 11.0), facecolor=SURFACE)
        gs = fig.add_gridspec(
            4, 6, height_ratios=[1.0, 0.9, 0.7, 0.5], hspace=0.55, wspace=0.3
        )
        t = data["time"]

        # Row 1 -- one small multiple per joint. Separate axes rather than one
        # overlay: the joints have different ranges, and small multiples avoid
        # needing six distinguishable hues.
        for joint_idx in range(6):
            ax = fig.add_subplot(gs[0, joint_idx])
            ax.plot(
                t,
                np.rad2deg(data["demo"][:, joint_idx]),
                color=COLOR_DEMO,
                linewidth=2.0,
                label="demo",
            )
            ax.plot(
                t,
                np.rad2deg(data["fitted"][:, joint_idx]),
                color=COLOR_FIT,
                linewidth=1.4,
                linestyle="--",
                label="fitted",
            )
            # Knot rug: where the fitter chose to spend capacity.
            ax.plot(
                data["knots"] / data["rate"],
                np.full(len(data["knots"]), ax.get_ylim()[0]),
                marker="|",
                linestyle="none",
                color=INK_MUTED,
                markersize=4,
                markeredgewidth=0.4,
                alpha=0.25,
            )
            self.style_axis(
                ax,
                ylabel="deg" if joint_idx == 0 else None,
                title=f"j{joint_idx} {JOINT_NAMES[joint_idx]}",
            )
            if joint_idx == 0:
                ax.legend(
                    loc="best",
                    fontsize=7,
                    frameon=False,
                    labelcolor=INK_MUTED,
                )

        # Row 2 -- residuals. The job here is magnitude against a threshold, not
        # identity, so all six joints share one muted stroke and only the worst
        # is named.
        ax = fig.add_subplot(gs[1, :])
        residual_deg = np.rad2deg(
            np.abs(data["fitted"][:, :6] - data["demo"][:, :6])
        )
        worst_joint = int(residual_deg.max(axis=0).argmax())
        for joint_idx in range(6):
            if joint_idx == worst_joint:
                continue
            ax.plot(t, residual_deg[:, joint_idx], color=INK_MUTED, linewidth=0.8, alpha=0.45)
        ax.plot(
            t,
            residual_deg[:, worst_joint],
            color=COLOR_FIT,
            linewidth=1.8,
            label=f"worst: j{worst_joint} {JOINT_NAMES[worst_joint]}",
        )
        ax.axhline(
            self.joint_tol_deg, color=INK, linewidth=1.2, linestyle=":",
            label=f"tolerance {self.joint_tol_deg:.4f}deg",
        )
        ax.set_ylim(0, max(self.joint_tol_deg * 1.35, residual_deg.max() * 1.15))
        self.style_axis(
            ax,
            ylabel="|error| [deg]",
            title="Joint reconstruction error (all 6 joints; grey = the other five)",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_MUTED)

        # Row 3 -- gripper on its own axes, never a second y-scale on a joint
        # plot: different units, and a dual axis invites false comparison.
        ax = fig.add_subplot(gs[2, :3])
        ax.plot(t, data["demo"][:, 6], color=COLOR_DEMO, linewidth=2.0, label="demo")
        ax.plot(
            t, data["fitted"][:, 6], color=COLOR_FIT, linewidth=1.4,
            linestyle="--", label="fitted",
        )
        self.style_axis(ax, ylabel="counts", xlabel="time [s]", title="Gripper")
        ax.legend(loc="best", fontsize=8, frameon=False, labelcolor=INK_MUTED)

        ax = fig.add_subplot(gs[2, 3:])
        gripper_err = np.abs(data["fitted"][:, 6] - data["demo"][:, 6])
        ax.plot(t, gripper_err, color=COLOR_FIT, linewidth=1.6)
        ax.axhline(
            self.gripper_tol_cnt, color=INK, linewidth=1.2, linestyle=":",
            label=f"tolerance {self.gripper_tol_cnt:.2f} cnt",
        )
        ax.set_ylim(0, max(self.gripper_tol_cnt * 1.35, gripper_err.max() * 1.15))
        self.style_axis(
            ax, ylabel="|error| [counts]", xlabel="time [s]",
            title="Gripper reconstruction error",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_MUTED)

        # Row 4 -- knot density: where the adaptive fitter spends capacity.
        ax = fig.add_subplot(gs[3, :])
        ax.hist(
            data["knots"] / data["rate"],
            bins=min(60, max(10, len(data["knots"]) // 2)),
            color=COLOR_DEMO,
            alpha=0.85,
        )
        self.style_axis(
            ax, ylabel="knots", xlabel="time [s]",
            title="Knot density -- peaks are where the trajectory bends",
        )

        joint_max = residual_deg.max()
        fig.suptitle(
            f"{data['name']}    "
            f"{data['n_frames']} frames / {t[-1]:.1f} s at {data['rate']:.1f} Hz    "
            f"{data['n_knots']} knots, compression {data['compression']:.2f}x    "
            f"joint max {joint_max:.4f}deg / {self.joint_tol_deg:.4f}deg    "
            f"gripper max {gripper_err.max():.2f} / {self.gripper_tol_cnt:.2f} cnt"
            + ("" if data["hit_tolerance"] else "    [TOLERANCE NOT REACHED]"),
            color=INK,
            fontsize=11,
            y=0.985,
        )
        return fig

    # ------------------------------------------------------- summary --------

    def plot_summary(self, all_data):
        import matplotlib.pylab as plt

        fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), facecolor=SURFACE)
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        compressions = np.array([d["compression"] for d in all_data])
        ax = axes[0, 0]
        ax.hist(compressions, bins=min(20, len(all_data)), color=COLOR_DEMO, alpha=0.85)
        ax.axvline(1.0, color=INK, linewidth=1.2, linestyle=":", label="no compression")
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        self.style_axis(
            ax, ylabel="episodes", xlabel="samples / knots",
            title=f"Compression  (median {np.median(compressions):.2f}x)",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_MUTED)

        ax = axes[0, 1]
        per_joint = np.array(
            [np.rad2deg(np.abs(d["fitted"][:, :6] - d["demo"][:, :6])).max(axis=0)
             for d in all_data]
        )
        ax.boxplot(
            [per_joint[:, j] for j in range(6)],
            tick_labels=[f"j{j}" for j in range(6)],
            medianprops=dict(color=COLOR_FIT, linewidth=1.8),
            boxprops=dict(color=INK_MUTED),
            whiskerprops=dict(color=INK_MUTED),
            capprops=dict(color=INK_MUTED),
            flierprops=dict(markeredgecolor=INK_MUTED, markersize=3),
        )
        ax.axhline(
            self.joint_tol_deg, color=INK, linewidth=1.2, linestyle=":",
            label=f"tolerance {self.joint_tol_deg:.4f}deg",
        )
        self.style_axis(
            ax, ylabel="max |error| [deg]",
            title="Per-joint reconstruction error across episodes",
        )
        ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_MUTED)

        ax = axes[1, 0]
        spans = np.concatenate([d["spans_s"] for d in all_data])
        ax.hist(spans, bins=40, color=COLOR_DEMO, alpha=0.85)
        ax.axvline(
            self.max_plan_age, color=INK, linewidth=1.2, linestyle=":",
            label=f"max_plan_age {self.max_plan_age:.2f}s",
        )
        self.style_axis(
            ax, ylabel="segments", xlabel="span [s] at 1x",
            title=f"Segment span  (median {np.median(spans):.2f}s -> replan "
            f"{1.0 / np.median(spans):.2f} Hz on exhaustion alone)",
        )
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_MUTED)

        ax = axes[1, 1]
        ax.scatter(
            [d["n_frames"] for d in all_data],
            [d["n_knots"] for d in all_data],
            color=COLOR_DEMO, s=28, alpha=0.85, edgecolor=SURFACE, linewidth=0.8,
        )
        self.style_axis(
            ax, ylabel="knots", xlabel="episode length [frames]",
            title="Knots vs length -- flat means knots follow shape, not duration",
        )

        n_fallback = sum(1 for d in all_data if not d["hit_tolerance"])
        fig.suptitle(
            f"{len(all_data)} episodes    max_error {self.max_error} "
            f"({self.joint_tol_deg:.4f}deg joints, {self.gripper_tol_cnt:.2f} cnt gripper)"
            + (f"    [{n_fallback} EPISODES MISSED TOLERANCE]" if n_fallback else ""),
            color=INK,
            fontsize=12,
        )
        return fig

    # ----------------------------------------------------------- driver -----

    def run(self):
        import matplotlib.pylab as plt

        name = self.__class__.__name__
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[{name}] Fitting {len(self.filenames)} episodes...")

        all_data = [self.fit_episode(f) for f in self.filenames]

        detail_idxes = (
            self.episode_idx
            if self.episode_idx
            else list(range(min(5, len(all_data))))
        )
        for episode_idx in detail_idxes:
            data = all_data[episode_idx]
            fig = self.plot_episode(data)
            out_path = os.path.join(self.output_dir, f"{data['name']}_fit.png")
            fig.savefig(out_path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
            print(f"[{name}] Saved {out_path}")
            if not self.show:
                plt.close(fig)

        if not self.no_summary and len(all_data) > 1:
            fig = self.plot_summary(all_data)
            out_path = os.path.join(self.output_dir, "dataset_summary.png")
            fig.savefig(out_path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
            print(f"[{name}] Saved {out_path}")
            if not self.show:
                plt.close(fig)

        compressions = np.array([d["compression"] for d in all_data])
        joint_max = max(
            np.rad2deg(np.abs(d["fitted"][:, :6] - d["demo"][:, :6])).max()
            for d in all_data
        )
        gripper_max = max(
            np.abs(d["fitted"][:, 6] - d["demo"][:, 6]).max() for d in all_data
        )
        fallback = [d["name"] for d in all_data if not d["hit_tolerance"]]
        print(
            f"[{name}] compression {compressions.min():.2f}-{compressions.max():.2f}x "
            f"(median {np.median(compressions):.2f}x)"
        )
        print(
            f"[{name}] worst joint {joint_max:.4f}/{self.joint_tol_deg:.4f} deg, "
            f"worst gripper {gripper_max:.2f}/{self.gripper_tol_cnt:.2f} cnt"
        )
        if fallback:
            print(f"[{name}] WARNING: tolerance not reached for {fallback}")

        if self.show:
            plt.show()


if __name__ == "__main__":
    PlotBsplineFit(**vars(parse_argument())).run()
