from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METHODS = ["raw", "lmsf", "sauvola", "basic", "sdf"]
LABELS = ["Raw", "LMSF", "Sauvola", "BaSiC", "ROSBD"]
COLORS = ["#64748b", "#e69f00", "#009e73", "#0072b2", "#d55e00"]
MARKERS = ["CD20", "CD1c", "CD206", "E-cadherin", "FXIIIa", "IL-10", "Podoplanin", "HepPar1"]
METRICS = ["balanced_accuracy", "auroc"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args()
    table = pd.read_csv(args.data_dir / "cellular_metrics.csv")
    if table.duplicated(["sample_id", "marker", "method"]).any():
        raise ValueError("Expected one row per core-marker-method")
    if len(table) != 160 or set(table.marker) != set(MARKERS) or set(table.method) != set(METHODS):
        raise ValueError("Expected four cores, eight markers, and five processing conditions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42,
        "axes.linewidth": .45, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": .45, "ytick.major.width": .45,
        "xtick.major.size": 2, "ytick.major.size": 2})
    fig = plt.figure(figsize=(468 / 72, 126 / 72))
    records = []
    for left, metric, title, letter in [(.055, "balanced_accuracy", "Balanced accuracy", "A"),
                                       (.555, "auroc", "AUROC", "B")]:
        ax = fig.add_axes([left, .27, .425, .55])
        for i, marker in enumerate(MARKERS):
            if i % 2 == 0:
                ax.axvspan(i - .49, i + .49, color="#f3f6f9", zorder=0)
            for j, (method, color) in enumerate(zip(METHODS, COLORS)):
                values = table.loc[table.marker.eq(marker) & table.method.eq(method), metric].dropna().to_numpy()
                expected = 2 if marker == "Podoplanin" else 4
                if len(values) != expected or not np.isfinite(values).all():
                    raise ValueError(f"Unexpected valid-core count: {marker}/{method}/{metric}")
                x = i + (j - 2) * .14
                ax.scatter(np.full(len(values), x), values, s=2.3, color=color, alpha=.32, linewidths=0, zorder=2)
                ax.scatter([x], [values.mean()], marker="D", s=6.5, color=color,
                           edgecolor="white", linewidth=.2, zorder=3)
                records.append(dict(marker=marker, method=method, metric=metric,
                                    n_valid_cores=len(values), mean=float(values.mean())))
        ax.axhline(.5, color="#9ca3af", lw=.5, ls="--")
        ax.set_xlim(-.5, 7.5); ax.set_ylim(.2, 1.02)
        ax.set_yticks([.2, .4, .6, .8, 1])
        ax.set_xticks(range(8), MARKERS, rotation=40, ha="right", fontsize=5.3)
        ax.tick_params(axis="y", labelsize=5.4, pad=1.5)
        ax.tick_params(axis="x", pad=1)
        ax.set_title(title, fontsize=7, pad=3)
        ax.grid(axis="y", color="#d9e1e8", lw=.35)
        ax.text(-.105, 1.075, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)
    handles = [Line2D([], [], marker="D", linestyle="None", markersize=3, color=c, label=l)
               for l, c in zip(LABELS, COLORS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.025), ncol=5,
               frameon=False, handletextpad=.25, columnspacing=1, fontsize=6)
    for extension in ["pdf", "png"]:
        fig.savefig(args.output_dir / f"fig6.{extension}", dpi=args.dpi)
    plt.close(fig)
    pd.DataFrame(records).to_csv(args.output_dir / "plotted_means.csv", index=False)
    macro = table.groupby("method")[METRICS].mean().reindex(METHODS)
    report = {"input_rows": len(table), "valid_pairs_per_method": 30,
              "aggregation": "equal-weight valid cores within each marker; equal-weight core-marker pairs for macro scores",
              "macro_scores": macro.to_dict(orient="index")}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output_dir / "fig6.pdf")


if __name__ == "__main__":
    main()
