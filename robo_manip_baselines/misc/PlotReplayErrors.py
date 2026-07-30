import argparse
import csv
import os
from collections import defaultdict

import matplotlib
import numpy as np

# Reference data-viz palette. Line charts use the adjacent pairlist, on which the
# documented hue order passes every gate, so up to four series is safe here.
MODE_COLORS = {
    "ur_rtde_raw": "#2a78d6",  # slot 1 blue
    "raw": "#eb6834",  # slot 2 orange
    "spline": "#1baf7a",  # slot 3 aqua
    "segments": "#eda100",  # slot 4 yellow
}
MODE_ORDER = ["ur_rtde_raw", "raw", "spline", "segments"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"
SURFACE = "#fcfcfb"

STAGES = ["fit", "execute", "track"]
STAGE_BLURB = {
    "fit": "spline representation loss (Algorithm 1)",
    "execute": "representation + seam stitching + clamp",
    "track": "controller: commanded vs measured",
}
METRICS = {"max": "Maximum", "rms": "RMS"}


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Plot joint error against speedup from replay logs. Reads the CSVs "
            "written by ReplayBsplineDemo.py --log and ReplayRealUR5eDemo.py "
            "--log, and writes one PNG per (metric, stage): six in total. "
            "Gripper columns are logged but not plotted."
        ),
    )
    parser.add_argument(
        "log",
        type=str,
        nargs="+",
        help="one or more replay-log CSVs (rows are pooled)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./replay_error_plots",
        help="directory to write PNGs into",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="display windows as well as saving (avoid over ssh -X)",
    )
    parser.add_argument(
        "--max_error", type=float, default=0.002,
        help="fit tolerance, drawn as a reference line on the fit plots",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="also write a single 2x3 overview figure",
    )
    return parser.parse_args()


class PlotReplayErrors:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not self.show:
            matplotlib.use("agg")
        self.joint_tol_deg = np.rad2deg(self.max_error)
        self.rows = self.load_rows()

    # --------------------------------------------------------------- data ----

    def load_rows(self):
        rows = []
        for path in self.log:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    rows.append(row)
        if not rows:
            raise ValueError("[PlotReplayErrors] No rows found in the given logs.")
        print(f"[{self.__class__.__name__}] Loaded {len(rows)} rows from {len(self.log)} file(s)")
        return rows

    def series_for(self, stage, metric):
        """``{mode: (speedups, [values per speedup])}`` for one stage+metric.

        Blank cells are skipped rather than read as zero: a script that cannot
        measure a stage leaves it empty, and treating that as no-error would be
        the wrong conclusion.
        """
        column = f"{stage}_joint_{metric}_deg"
        grouped = defaultdict(lambda: defaultdict(list))
        for row in self.rows:
            value = row.get(column, "")
            if value in ("", None):
                continue
            try:
                grouped[row["mode"]][float(row["speedup"])].append(float(value))
            except (TypeError, ValueError):
                continue

        series = {}
        for mode, by_speedup in grouped.items():
            speedups = sorted(by_speedup)
            series[mode] = (
                np.array(speedups),
                [np.array(by_speedup[s]) for s in speedups],
            )
        return series

    # -------------------------------------------------------------- style ----

    @staticmethod
    def style_axis(ax, title, subtitle, ylabel):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)
        ax.set_xlabel("speedup [x]", color=INK_MUTED, fontsize=10)
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
        # Title padded clear of the axes, subtitle tucked just under it.
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=24)
        if subtitle:
            ax.text(
                0.0, 1.012, subtitle, transform=ax.transAxes,
                color=INK_MUTED, fontsize=9, ha="left", va="bottom",
            )

    def draw(self, ax, stage, metric):
        series = self.series_for(stage, metric)
        if not series:
            ax.text(
                0.5, 0.5, f"no {stage} data in log", transform=ax.transAxes,
                ha="center", va="center", color=INK_MUTED, fontsize=11,
            )
            self.style_axis(
                ax, f"{METRICS[metric]} joint error -- {stage}",
                STAGE_BLURB[stage], "error [deg]",
            )
            return 0

        ordered = [m for m in MODE_ORDER if m in series]
        ordered += [m for m in sorted(series) if m not in MODE_ORDER]

        for mode in ordered:
            speedups, values = series[mode]
            medians = np.array([v.mean() for v in values])
            color = MODE_COLORS.get(mode, INK_MUTED)
            # Range band only where a speedup has more than one episode.
            if any(len(v) > 1 for v in values):
                ax.fill_between(
                    speedups,
                    [v.min() for v in values],
                    [v.max() for v in values],
                    color=color, alpha=0.15, linewidth=0,
                )
            ax.plot(
                speedups, medians, color=color, linewidth=2.0,
                marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=mode,
            )

        if stage == "fit":
            ax.axhline(
                self.joint_tol_deg, color=INK, linewidth=1.2, linestyle=":",
                label=f"fit tolerance {self.joint_tol_deg:.4f}deg",
            )

        recorded = self.series_for("recorded", metric)
        if stage == "track" and recorded:
            level = float(
                np.mean([v.mean() for vals in recorded.values() for v in vals[1]])
            )
            ax.axhline(
                level, color=INK, linewidth=1.2, linestyle=":",
                label=f"recording's own tracking {level:.2f}deg",
            )

        self.style_axis(
            ax, f"{METRICS[metric]} joint error -- {stage}",
            STAGE_BLURB[stage], "error [deg]",
        )
        ax.legend(fontsize=9, frameon=False, labelcolor=INK_MUTED, loc="best")
        return len(ordered)

    # ------------------------------------------------------------- driver ----

    def run(self):
        import matplotlib.pylab as plt

        name = self.__class__.__name__
        os.makedirs(self.output_dir, exist_ok=True)

        for metric in METRICS:
            for stage in STAGES:
                fig, ax = plt.subplots(figsize=(7.0, 4.8), facecolor=SURFACE)
                n_series = self.draw(ax, stage, metric)
                fig.tight_layout()
                out_path = os.path.join(
                    self.output_dir, f"joint_{metric}_{stage}_vs_speedup.png"
                )
                fig.savefig(out_path, dpi=120, facecolor=SURFACE, bbox_inches="tight")
                print(f"[{name}] Saved {out_path} ({n_series} series)")
                if not self.show:
                    plt.close(fig)

        if self.combined:
            fig, axes = plt.subplots(2, 3, figsize=(19.0, 9.0), facecolor=SURFACE)
            for row_idx, metric in enumerate(METRICS):
                for col_idx, stage in enumerate(STAGES):
                    self.draw(axes[row_idx, col_idx], stage, metric)
            fig.tight_layout()
            out_path = os.path.join(self.output_dir, "overview.png")
            fig.savefig(out_path, dpi=110, facecolor=SURFACE, bbox_inches="tight")
            print(f"[{name}] Saved {out_path}")
            if not self.show:
                plt.close(fig)

        self.print_table()

        if self.show:
            plt.show()

    def print_table(self):
        name = self.__class__.__name__
        print(f"\n[{name}] joint error [deg], mean over episodes")
        header = f"{'mode':<14}{'speedup':>9}"
        for stage in STAGES:
            header += f"{stage + ' max':>12}{stage + ' rms':>12}"
        print(f"[{name}] {header}")

        combos = sorted(
            {(row["mode"], float(row["speedup"])) for row in self.rows},
            key=lambda c: (MODE_ORDER.index(c[0]) if c[0] in MODE_ORDER else 99, c[1]),
        )
        for mode, speedup in combos:
            line = f"{mode:<14}{speedup:>9.2f}"
            for stage in STAGES:
                for metric in ("max", "rms"):
                    vals = [
                        float(r[f"{stage}_joint_{metric}_deg"])
                        for r in self.rows
                        if r["mode"] == mode
                        and float(r["speedup"]) == speedup
                        and r.get(f"{stage}_joint_{metric}_deg", "") not in ("", None)
                    ]
                    line += f"{np.mean(vals):>12.4f}" if vals else f"{'-':>12}"
            print(f"[{name}] {line}")


if __name__ == "__main__":
    PlotReplayErrors(**vars(parse_argument())).run()
