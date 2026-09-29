"""Recreate the two existing paper figures from authorized data and local results."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from rpmlt.data import load_data
from rpmlt.features import ASSETS
from rpmlt.evaluation import WINDOWS


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .5, "axes.axisbelow": True,
                         "grid.color": "#e3e7eb", "grid.linewidth": .5})


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=900)
    plt.close(fig)


def data_context(data, output):
    totals = pd.DataFrame({"Within-cell": data.X[:, data.is_diag].sum(1),
                           "Between-cell": data.X[:, ~data.is_diag].sum(1)}, index=data.dates)
    normalizer = totals.loc["2023-11-01":"2023-12-31"].mean()
    totals = (totals / normalizer).reindex(pd.date_range("2023-11-01", "2024-10-31"))
    water = pd.read_csv(ASSETS / "nanao_water.csv", parse_dates=["date"])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2))
    fig.subplots_adjust(left=.09, right=.99, bottom=.19, top=.85, wspace=.4)
    for col, color, marker in zip(totals.columns, ["#187b80", "#c16a32"], ["o", "s"]):
        axes[0].plot(totals.index, totals[col], lw=.8, color=color, label=col, marker=marker, markevery=20, ms=2)
    axes[0].axvline(pd.Timestamp("2024-01-01"), color="#737a83", ls="--", lw=.8)
    axes[0].set(ylabel="Daily total / pre-earthquake mean", title="(a) Mobility observations")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[11, 1, 3, 5, 7, 9]))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for col, label, color, marker in [("supplied", "Flow restored", "#187b80", "o"),
                                     ("drink_supplied", "Drinkable supply", "#c16a32", "s")]:
        axes[1].step(water.date, water[col] / water.households, where="post", color=color, lw=1.2, label=label)
        axes[1].scatter(water.date, water[col] / water.households, s=7, color=color, marker=marker)
    overlap = data.dates[(data.dates >= water.date.min()) & (data.dates <= water.date.max())]
    axes[1].plot(overlap, np.full(len(overlap), -.065), "|", color="#49515b", ms=5)
    axes[1].set(xlim=(pd.Timestamp("2024-01-20"), pd.Timestamp("2024-04-08")), ylim=(-.1, 1.04),
                ylabel="Fraction of reported households", title="(b) Nanao water-service restoration")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b 2024"))
    for ax in axes:
        ax.legend(frameon=False, loc="upper left" if ax is axes[1] else "upper right")
        ax.axvspan(pd.Timestamp("2024-02-01"), pd.Timestamp("2024-04-01"), color="#e9edf3", zorder=0)
        ax.grid(axis="y")
        ax.tick_params(direction="in")
    save(fig, output / "data_context")


def errors(evaluation, output):
    scores = pd.read_csv(evaluation / "scores.csv")
    pivot = scores.pivot(index="window", columns="model", values="combined").loc[[w.name for w in WINDOWS]]
    improvement = 100 * (pivot.local - pivot.combined) / pivot.local
    labels = ["Jan 8-31", "Jan 15-31", "Jan 22-31", "Apr 1-30", "Apr 1-14", "May 1-Jun 29",
              "Jul 1-Aug 29", "Jul 6-Aug 3", "Sep 1-Oct 30", "Dec 6-31 (2023)"]
    selected = pd.read_csv(evaluation / "selected_errors.csv").pivot(index="pair", columns="model", values="rmse")
    selected = selected.loc[["40_46_40_46", "54_56_54_56", "41_47_41_47", "58_44_58_44"]]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.65), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=.15, right=.99, bottom=.24, top=.85, wspace=.45)
    axes[0].barh(np.arange(10), improvement, color="#187b80", height=.65)
    axes[0].set_yticks(np.arange(10), labels)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Score reduction vs. local-level (%)", title="(a) Held-out improvement")
    for offset, name, label, color, hatch in [(-.24, "local", "Local", "#8393a6", ""),
                                             (0, "temporal", "Temporal", "#c16a32", "//"),
                                             (.24, "combined", "Combined", "#187b80", "..")]:
        axes[1].bar(np.arange(4) + offset, selected[name], width=.22, label=label, color=color, hatch=hatch)
    axes[1].set_xticks(np.arange(4), ["Nanao\n(40,46)", "Noto\n(54,56)", "Nanao\n(41,47)", "Wajima\n(58,44)"])
    axes[1].set(ylabel="Diagonal-grid RMSE", ylim=(0, 78), title="(b) Selected spatial errors")
    axes[1].legend(frameon=False, ncol=3, columnspacing=.6, handlelength=1)
    for ax in axes:
        ax.grid(axis="x" if ax is axes[0] else "y")
        ax.tick_params(direction="in")
    save(fig, output / "badcase_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    data_context(load_data(args.data), args.output)
    errors(args.evaluation, args.output)
