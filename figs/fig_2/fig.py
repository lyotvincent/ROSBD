from pathlib import Path
from io import BytesIO
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

HERE = Path(__file__).resolve().parent
ORDER = ["sdf", "clearaf", "sauvola", "lmsf", "basic"]
LABELS = dict(sdf="ROSBD", clearaf="CLEAR-AF", sauvola="Sauvola", lmsf="LMSF", basic="BaSiC")
COLORS = dict(sdf="#246C86", clearaf="#BB843F", sauvola="#969CA3", lmsf="#8D7BA8", basic="#6D9F8A")
N = dict(orion=61, cycif=37, bmcycif=3)
DATASETS = dict(orion="CRC Orion", cycif="CRC CyCIF", bmcycif="Breast TMA")
CELL_ORDER = ["raw", "lmsf", "sauvola", "basic", "sdf"]
CELL_METRICS = ["accuracy", "positive_recall", "negative_recall", "balanced_accuracy", "auroc"]


def load_data(directory):
    ssim = pd.read_csv(directory / "bg_ssim_sample.csv")
    psnr = pd.read_csv(directory / "result_PSNR_sample.csv")
    cells = pd.read_csv(directory / "cellular_metrics.csv")
    reported = pd.read_csv(directory / "cellular_macro_summary.csv").set_index("method")
    for frame, metric in [(ssim, "ssim"), (psnr, "psnr_db")]:
        if frame.duplicated(["dataset", "method", "sample_id"]).any():
            raise ValueError("Image-level inputs must contain one row per image set and method")
        if not np.isfinite(frame[metric]).all():
            raise ValueError(f"Nonfinite {metric} in the frozen data")
        for (dataset, method), group in frame.groupby(["dataset", "method"]):
            if len(group) != N[dataset]:
                raise ValueError(f"Unexpected sample count: {dataset}/{method}")
    if cells.duplicated(["sample_id", "marker", "method"]).any():
        raise ValueError("Cellular inputs must contain one row per core-marker-method")
    macro = cells.groupby("method")[CELL_METRICS].mean().reindex(CELL_ORDER)
    np.testing.assert_allclose(macro, reported.loc[CELL_ORDER, CELL_METRICS], rtol=1e-12, atol=1e-12)
    counts = cells.groupby("method")[CELL_METRICS].count().reindex(CELL_ORDER)
    np.testing.assert_array_equal(counts.to_numpy(), np.tile([32, 30, 32, 30, 30], (5, 1)))
    return ssim, psnr, macro, counts


def draw_distributions(frame, metric, output, dpi):
    """Recreate the original vector boxplots, with footer text kept in the caption."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False})
    datasets = ["orion", "cycif"] if metric == "ssim" else ["orion", "cycif", "bmcycif"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5, 3.15), squeeze=False)
    for index, dataset in enumerate(datasets):
        ax = axes[0, index]
        part = frame.loc[frame.dataset.eq(dataset)]
        methods = [m for m in ORDER if m in set(part.method)]
        values = [part.loc[part.method.eq(m), metric].to_numpy() for m in methods]
        boxes = ax.boxplot(values, positions=range(len(methods)), widths=.58,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="#17252B", linewidth=1.15),
            whiskerprops=dict(linewidth=.8), capprops=dict(linewidth=.8))
        rng = np.random.default_rng(20261024 + index)
        for i, (method, data) in enumerate(zip(methods, values)):
            boxes["boxes"][i].set(facecolor=COLORS[method], edgecolor=COLORS[method], alpha=.30)
            ax.scatter(i + rng.uniform(-.14, .14, len(data)), data,
                s=8 if dataset != "bmcycif" else 15, alpha=.65, c=COLORS[method], linewidths=0, zorder=3)
        ax.set_xticks(range(len(methods)), [LABELS[m] + ("\u2020" if m == "clearaf" else "") for m in methods],
                      rotation=38, ha="right")
        for label, method in zip(ax.get_xticklabels(), methods):
            if method == "sdf":
                label.set_color(COLORS[method]); label.set_fontweight("bold")
        ax.set_title(f"({chr(97 + index)}) {DATASETS[dataset]} ($n={N[dataset]}$)", loc="left")
        ax.grid(axis="y", color="#DCE1E5", linewidth=.55)
        ax.set_axisbelow(True)
        if metric == "ssim":
            ax.set_ylim(.30, 1.015)
        ax.set_ylabel(("Background SSIM" if metric == "ssim" else "Raw-reference PSNR (dB)") if index == 0 else "")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.subplots_adjust(left=.085, right=.99, bottom=.32, top=.89, wspace=.35)
    raw_pdf = BytesIO()
    fig.savefig(raw_pdf, format="pdf", dpi=dpi)
    plt.close(fig)
    raw_pdf.seek(0)
    page = PdfReader(raw_pdf).pages[0]
    # The manuscript removes the 37.36-pt footer band; all axes and data remain intact.
    page.mediabox.lower_left = (0, 37.36)
    page.cropbox.lower_left = (0, 37.36)
    writer = PdfWriter(); writer.add_page(page)
    with output.open("wb") as handle:
        writer.write(handle)
    return output


def image_summary(ssim, psnr):
    def value(frame, dataset, method, metric, digits):
        x = frame.loc[frame.dataset.eq(dataset) & frame.method.eq(method), metric]
        return f"{x.mean():.{digits}f} ± {x.std(ddof=1):.{digits}f}"
    rows = []
    for method in ORDER:
        rows.append(["ROSBD (ours)" if method == "sdf" else LABELS[method],
            "Required" if method == "clearaf" else "None",
            *[value(ssim, ds, method, "ssim", 4) for ds in ["orion", "cycif"]],
            *["—" if ds == "bmcycif" and method == "clearaf" else value(psnr, ds, method, "psnr_db", 2)
              for ds in ["orion", "cycif", "bmcycif"]]])
    return rows


def draw_summary_tables(image_rows, macro, output):
    width, height = 468, 560
    c = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    for letter, y, title in [("A", 548, "Image-level score distributions"),
                              ("B", 168, "Image-level summary (mean ± SD)"),
                              ("C", 78, "Cellular marker classification")]:
        c.setFont("Helvetica-Bold", 12); c.drawString(0, y, letter)
        c.setFont("Helvetica", 9); c.drawString(16, y + .5, title)
    common = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1), ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 0), (-1, 0), .7, colors.black),
        ("LINEBELOW", (0, 1), (-1, 1), .5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .7, colors.black)]
    data = [["Method", "AF ref.", "Background SSIM", "", "Raw-reference PSNR (dB)", "", ""],
            ["", "", "Orion", "CyCIF", "Orion", "CyCIF", "Breast TMA"], *image_rows]
    table = Table(data, colWidths=[82, 41, 65, 65, 61, 61, 65], rowHeights=[11, 11] + [10] * 5)
    table.setStyle(TableStyle(common + [("SPAN", (0, 0), (0, 1)), ("SPAN", (1, 0), (1, 1)),
        ("SPAN", (2, 0), (3, 0)), ("SPAN", (4, 0), (6, 0)),
        ("LINEBELOW", (2, 0), (3, 0), .4, colors.black), ("LINEBELOW", (4, 0), (6, 0), .4, colors.black),
        ("FONTNAME", (0, 2), (0, 2), "Helvetica-Bold"), ("FONTNAME", (2, 2), (3, 2), "Helvetica-Bold")]))
    _, h = table.wrap(width, height); table.drawOn(c, 14, 162 - h)
    cell_rows = []
    for method in CELL_ORDER:
        cell_rows.append(["Raw" if method == "raw" else "ROSBD (ours)" if method == "sdf" else LABELS[method],
            *[f"{macro.loc[method, metric] * (1 if metric == 'auroc' else 100):.{4 if metric == 'auroc' else 2}f}"
              for metric in CELL_METRICS]])
    data = [["Method", "Accuracy", "Positive", "Negative", "Balanced", "AUROC"],
            ["", "(%)", "recall (%)", "recall (%)", "accuracy (%)", ""], *cell_rows]
    style = common + [("SPAN", (0, 0), (0, 1)), ("SPAN", (5, 0), (5, 1)),
                      ("FONTNAME", (0, 6), (0, 6), "Helvetica-Bold")]
    for column, metric in enumerate(CELL_METRICS, 1):
        best = macro[metric].max()
        for row, method in enumerate(CELL_ORDER, 2):
            if np.isclose(macro.loc[method, metric], best, rtol=0, atol=1e-14):
                style.append(("FONTNAME", (column, row), (column, row), "Helvetica-Bold"))
    table = Table(data, colWidths=[108, 62, 68, 68, 78, 56], rowHeights=[10] * 7)
    table.setStyle(TableStyle(style))
    _, h = table.wrap(width, height); table.drawOn(c, 14, 72 - h)
    c.showPage(); c.save()
    return cell_rows


def assemble(plots, overlay, output):
    writer = PdfWriter(); page = writer.add_blank_page(468, 560)
    plot_width = 440
    plot_height = 189.44 * plot_width / 468
    for path, top in zip(plots, [538, 538 - plot_height - 2]):
        source = PdfReader(path).pages[0]
        scale = plot_width / float(source.mediabox.width)
        transform = Transformation().translate(-float(source.mediabox.left), -float(source.mediabox.bottom))
        page.merge_transformed_page(source, transform.scale(scale).translate(14, top - float(source.mediabox.height) * scale))
    page.merge_page(PdfReader(overlay).pages[0])
    writer.add_metadata({"/Title": "Figure 2: ROSBD quantitative comparison"})
    with output.open("wb") as handle:
        writer.write(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--dpi", type=int, default=350)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    ssim, psnr, macro, counts = load_data(args.data_dir)
    plots = [draw_distributions(ssim, "ssim", args.output_dir / "background_ssim_boxplot.pdf", args.dpi),
             draw_distributions(psnr, "psnr_db", args.output_dir / "raw_reference_psnr_boxplot.pdf", args.dpi)]
    overlay = args.output_dir / "summary_panels.pdf"
    rows = image_summary(ssim, psnr)
    cell_rows = draw_summary_tables(rows, macro, overlay)
    output = args.output_dir / "fig2.pdf"
    assemble(plots, overlay, output)
    text = PdfReader(output).pages[0].extract_text()
    for row in rows + cell_rows:
        for field in row:
            for number in re.findall(r"\d+\.\d+", field):
                if number not in text:
                    raise ValueError(f"Missing displayed value: {number}")
    report = {"image_level_rows": {"ssim": len(ssim), "psnr": len(psnr)},
        "image_level_display": rows, "cellular_display": cell_rows,
        "cellular_valid_pairs": counts.to_dict(orient="index")}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
