"""Observed spatial heterogeneity, without predictions or manuscript edits.

Run from any working directory. Inputs are read-only. All outputs remain here.
Geography: MLIT N03, 2024-01-01 (CC BY 4.0), Ishikawa and Toyama.
"""
from pathlib import Path
import ast
import argparse
import io
import json
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import pandas as pd

HERE = Path("outputs/spatial")
DATAFILE = None
width_inch = 7.0
Fontsize = 8
ax_linewidth = 0.5
DPI = 900
plt.rcParams.update({
    # User-requested match to the manuscript's existing figures overrides Arial.
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": Fontsize, "axes.linewidth": ax_linewidth,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white",
})
DLON = (138.042 - 136.029) / 99
DLAT = (37.646 - 36.203) / 69
EXTENT = (136.029 + 28.5 * DLON, 136.029 + 69.5 * DLON,
          36.203 + 33.5 * DLAT, 36.203 + 69.5 * DLAT)
PERIODS = [("baseline", "2023-11-01", "2023-12-31"),
           ("january", "2024-01-08", "2024-01-31"),
           ("april", "2024-04-01", "2024-04-30")]


def load_observations():
    from rpmlt.data import load_data
    data = load_data(DATAFILE)
    dates = data.dates
    pairs = np.asarray(data.pairs)[data.is_diag][:, :2]
    values = data.X[:, data.is_diag].astype(float)
    assert len(dates) == 292 and len(pairs) == 922
    assert not ((dates >= "2024-02-01") & (dates <= "2024-03-31")).any()
    grid = pd.MultiIndex.from_product([range(35, 71), range(30, 71)],
                                     names=["cell_y", "cell_x"])
    table = pd.DataFrame(index=grid)
    counts = {}
    for name, start, end in PERIODS:
        mask = (dates >= start) & (dates <= end)
        means = values[mask].mean(axis=0)
        table[name] = pd.Series(means, index=pd.MultiIndex.from_arrays(pairs.T))
        table[name] = table[name].fillna(0.)
        counts[name] = int(mask.sum())
    table["eligible"] = table.baseline >= 1.
    for name in ["january", "april"]:
        table[name + "_change_pct"] = np.where(
            table.eligible, 100 * (table[name] / table.baseline.replace(0, np.nan) - 1), np.nan)
    # Independent reconciliation with source TSV, not just cached arrays.
    raw = {name: [] for name, _, _ in PERIODS}
    for line in DATAFILE.open(encoding="utf-8"):
        date, payload = line.rstrip().split("\t", 1)
        stamp = pd.to_datetime(date, format="%Y%m%d")
        selected = [name for name, start, end in PERIODS
                    if pd.Timestamp(start) <= stamp <= pd.Timestamp(end)]
        if not selected or payload.strip() == "NA":
            continue
        record = ast.literal_eval(payload)
        raw[selected[0]].append([float(record.get(f"{y}_{x}", {}).get(f"{y}_{x}", 0))
                                 for y, x in grid])
    for name, _, _ in PERIODS:
        assert len(raw[name]) == counts[name]
        np.testing.assert_allclose(np.mean(raw[name], axis=0), table[name], rtol=1e-6, atol=1e-6)
    return table, counts


def load_boundaries():
    """Fetch official geometry only; never fetch any additional mobility data."""
    cache = HERE / "source_geometry"
    cache.mkdir(exist_ok=True)
    polygons, source_urls = [], []
    for pref in ["17", "16"]:
        url = f"https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2024/N03-20240101_{pref}_GML.zip"
        source_urls.append(url)
        path = cache / f"N03_2024_{pref}.geojson"
        if not path.exists():
            print(f"Downloading MLIT prefecture {pref}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                names = [n for n in archive.namelist() if n.endswith(".geojson")]
                assert len(names) == 1
                path.write_bytes(archive.read(names[0]))
        data = json.loads(path.read_text(encoding="utf-8"))
        for feature in data["features"]:
            geom = feature["geometry"]
            parts = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            for part in parts:
                ring = np.asarray(part[0])[:, :2]
                if (ring[:, 0].max() >= EXTENT[0] and ring[:, 0].min() <= EXTENT[1]
                        and ring[:, 1].max() >= EXTENT[2] and ring[:, 1].min() <= EXTENT[3]):
                    polygons.append(ring)
    return polygons, source_urls


def style_map(ax, boundaries, show_y):
    ax.set_facecolor("#ffffff")
    ax.add_collection(PolyCollection(boundaries, facecolors="#e5e8eb", edgecolors="none", zorder=0))
    ax.set_xlim(*EXTENT[:2])
    ax.set_ylim(*EXTENT[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad(37.28)))
    ax.set_xticks([136.7, 137.0, 137.3], ["136.7°E", "137.0°E", "137.3°E"])
    ax.set_yticks([37.0, 37.2, 37.4, 37.6], ["37.0°N", "37.2°N", "37.4°N", "37.6°N"])
    # Geographic extents follow the scoring box; ticks use round degree values.
    ax.tick_params(which="both", direction="in", width=ax_linewidth, length=2.5,
                   labelleft=show_y, left=show_y, pad=3)
    for name in ["top", "right"]:
        ax.spines[name].set_visible(False)
    for name in ["bottom", "left"]:
        ax.spines[name].set_color("#69727b")
        ax.spines[name].set_linewidth(ax_linewidth)


def build_figure(table, counts, boundaries):
    figure_number = 1  # One full-width compound figure, with three map panels.
    linewidth = 0.35
    Color = ["#a94b24", "#f8f7f4", "#226e9c"]
    # Continuous heatmaps encode one scalar, not categorical colored series.
    Markers = ["o"]
    diverging = LinearSegmentedColormap.from_list("orange_white_blue", Color)
    fig, axes = plt.subplots(1, 3, dpi=180)
    fig.set_size_inches(width_inch / figure_number, 3.4)
    fig.subplots_adjust(left=.067, right=.989, bottom=.24, top=.86, wspace=.065)
    xedges = np.linspace(EXTENT[0], EXTENT[1], 42)
    yedges = np.linspace(EXTENT[2], EXTENT[3], 37)
    subtitles = [f"Nov-Dec 2023 | {counts['baseline']} observed days",
                 f"8-31 Jan 2024 | {counts['january']} observed days",
                 f"1-30 Apr 2024 | {counts['april']} observed days"]
    titles = ["(a) pre-earthquake activity", "(b) early post-earthquake", "(c) after the missing interval"]
    for k, ax in enumerate(axes):
        style_map(ax, boundaries, k == 0)
        column = ["baseline", "january_change_pct", "april_change_pct"][k]
        matrix = table[column].to_numpy().reshape(36, 41)
        if k == 0:
            matrix = np.ma.masked_less_equal(matrix, 0)
            norm, cmap = LogNorm(vmin=.1, vmax=1000), "viridis"
        else:
            matrix = np.ma.masked_invalid(matrix)
            norm, cmap = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100), diverging
        mesh = ax.pcolormesh(xedges, yedges, matrix, norm=norm, cmap=cmap,
                             shading="flat", edgecolors="none", zorder=2)
        ax.add_collection(LineCollection(boundaries, colors="#606a70", linewidths=linewidth,
                                         alpha=.85, zorder=3))
        ax.set_title(titles[k], loc="left", fontsize=9, weight="normal", pad=24)
        ax.text(0, 1.025, subtitles[k], transform=ax.transAxes, fontsize=7, va="bottom")
        if k == 0:
            baseline_mesh = mesh
        else:
            change_mesh = mesh
    # Labels identify regions, not exact municipal offices or selected samples.
    for label, point in [("Wajima", (136.84, 37.46)), ("Suzu", (137.33, 37.535)),
                         ("Noto", (137.21, 37.27)), ("Nanao", (137.10, 37.03))]:
        axes[0].text(*point, label, ha="center", fontsize=7, color="#202b33", zorder=5,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    # The two cells already discussed in the manuscript, not post-hoc extrema.
    for ax, period in zip(axes[1:], ["january", "april"]):
        for label, cell, label_pos in [
            ("Wajima", (58, 44), (136.74, 37.50)),
            ("Nanao", (41, 47), (137.24, 37.115)),
        ]:
            yy, xx = cell
            xy = (136.029 + (xx - 1) * DLON, 36.203 + (yy - 1) * DLAT)
            change = table.loc[cell, period + "_change_pct"]
            ax.plot(*xy, marker=Markers[0], ms=4, mfc="none", mec="#243843", mew=.7, zorder=6)
            ax.annotate(f"{label}\n{change:+.1f}%", xy=xy, xytext=label_pos,
                        ha="center", va="center", fontsize=7, color="#243843", zorder=7,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": .95},
                        arrowprops={"arrowstyle": "-", "lw": .6, "color": "#243843", "shrinkB": 3})
    # True 10 km at the scale-bar latitude under the local geographic aspect.
    lat = 36.968
    span = 10 / (111.32 * np.cos(np.deg2rad(lat)))
    axes[0].plot([136.65, 136.65 + span], [lat, lat], color="#27313a", lw=1.5, zorder=5)
    axes[0].text(136.65 + span / 2, lat + .014, "10 km", ha="center", fontsize=7, zorder=5)
    axes[0].annotate("N", xy=(136.66, 37.60), xytext=(136.66, 37.53), ha="center", fontsize=8,
                     arrowprops={"arrowstyle": "-|>", "lw": .7, "color": "#27313a"})
    cax0 = fig.add_axes([.085, .12, .248, .026])
    cbar0 = fig.colorbar(baseline_mesh, cax=cax0, orientation="horizontal", extend="min")
    cbar0.set_ticks([.1, 1, 10, 100, 1000], labels=["0.1", "1", "10", "100", "1,000"])
    cbar0.set_label("Mean daily within-cell activity (log scale)", fontsize=7, labelpad=3)
    cax1 = fig.add_axes([.428, .12, .505, .026])
    cbar1 = fig.colorbar(change_mesh, cax=cax1, orientation="horizontal", extend="max")
    cbar1.set_ticks([-100, -50, 0, 50, 100], labels=["-100%", "-50%", "0", "+50%", "+100%"])
    cbar1.set_label("Change from each cell's Nov-Dec mean (shared scale)", fontsize=7, labelpad=3)
    for cbar in [cbar0, cbar1]:
        cbar.outline.set_linewidth(.4)
        cbar.ax.tick_params(direction="in", labelsize=7, length=2, width=.4)
    fig.text(.66, .179, "February-March: no observations; no interpolation shown",
             fontsize=7, color="#48545e", ha="center")
    fig.text(.067, .009, "Boundaries: MLIT N03 (2024), CC BY 4.0.",
             fontsize=6.8, color="#48545e")
    fig.canvas.draw()
    # Detect text clipping instead of cropping the exported physical dimensions.
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text() and text.get_window_extent(renderer).width > 0:
            box = text.get_window_extent(renderer)
            # Tick locators may retain invisible out-of-view children.
            if text in fig.texts:
                assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(HERE / f"spatial_evolution.{ext}", dpi=DPI)
    plt.close(fig)


def write_evidence(table, counts, urls):
    selected = table[table.eligible]
    summary = {
        "valid_days": counts,
        "eligible_cells": int(table.eligible.sum()),
        "total_scoring_cells": len(table),
        "baseline_activity_coverage": float(selected.baseline.sum() / table.baseline.sum()),
        "baseline_positive_cells": int((table.baseline > 0).sum()),
        "source_tsv_reconciliation": "passed for all cells and all three periods",
        "geometry_sources": urls,
        "source_documentation": "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-N03-2024.html",
        "map_width_inches": width_inch,
        "map_height_inches": 3.4,
        "raster_dpi": DPI,
        "sample_cells": {},
        "ratio_threshold_sensitivity": {},
    }
    for name in ["january", "april"]:
        s = selected[name + "_change_pct"]
        summary[name] = {"cell_change_percentiles": dict(zip(["p10", "p25", "p50", "p75", "p90"],
                          map(float, np.percentile(s, [10, 25, 50, 75, 90])))),
                         "above_100_percent": int((s > 100).sum()),
                         "below_minus_100_percent": int((s < -100).sum())}
    for threshold in [.5, 1., 2.]:
        subset = table[table.baseline >= threshold]
        summary["ratio_threshold_sensitivity"][str(threshold)] = {
            "cells": len(subset), "baseline_coverage": float(subset.baseline.sum()/table.baseline.sum()),
            **{f"{name}_median_pct": float(np.median(100*(subset[name]/subset.baseline-1)))
               for name in ["january", "april"]}}
    for cell in [(58, 44), (40, 46), (54, 56), (41, 47)]:
        summary["sample_cells"][f"{cell[0]}_{cell[1]}"] = {
            name: float(table.loc[cell, name])
            for name in ["baseline", "january_change_pct", "april_change_pct"]}
    table.reset_index().to_csv(HERE / "spatial_values.csv", index=False, float_format="%.8f")
    (HERE / "evidence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw observed spatial changes; requires authorized OD data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    DATAFILE, HERE = args.data, args.output
    HERE.mkdir(parents=True, exist_ok=True)
    table, counts = load_observations()
    boundaries, urls = load_boundaries()
    write_evidence(table, counts, urls)
    build_figure(table, counts, boundaries)
