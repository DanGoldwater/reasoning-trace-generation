"""Generate the EDA figures for docs/eda_report.md.

Run with: uv run --with matplotlib python docs/make_plots.py
Figures are written to docs/figures/ in light and dark variants.
"""

import collections
import json
import math
from pathlib import Path
from typing import Literal

import matplotlib
from matplotlib.axes import Axes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_PATH = Path("data/private_qa.json")
FIG_DIR = Path("docs/figures")
THRESHOLD = 0.1  # micromolar: observed sensitive cut-off

# Values taken unchanged from the dataviz reference palette.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "yes": "#2a78d6",
        "no": "#d66c2a",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "yes": "#3987e5",
        "no": "#f08a4b",
    },
}


def style(theme: dict[str, str]) -> None:
    """Apply the shared visual language to a figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": theme["primary"],
            "axes.labelcolor": theme["secondary"],
            "axes.edgecolor": theme["axis"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "grid.color": theme["grid"],
            "grid.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.size": 9,
        }
    )


def recessive(
    ax: Axes,
    theme: dict[str, str],
    axis: Literal["both", "x", "y"] = "x",
) -> None:
    """Add an understated grid behind the plotted marks."""
    ax.grid(axis=axis, linestyle="-", linewidth=0.8, color=theme["grid"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def log_ticks(ax: Axes) -> None:
    """Use powers of ten as labels for log-transformed IC50 values."""
    ticks = list(range(-6, 8))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"$10^{{{tick}}}$" for tick in ticks])


def threshold_line(ax: Axes, theme: dict[str, str]) -> None:
    """Mark the IC50 decision boundary."""
    x = math.log10(THRESHOLD)
    ax.axvline(x, color=theme["secondary"], lw=1.2, ls=(0, (4, 3)), zorder=5)
    ax.text(
        x + 0.12,
        ax.get_ylim()[1],
        "0.1 µM cut-off",
        color=theme["secondary"],
        rotation=90,
        ha="left",
        va="top",
        fontsize=8.5,
    )


def label_values(records: list[dict], label: str) -> list[float]:
    """Return log10 IC50 values for one outcome label."""
    return [
        math.log10(record["metadata"]["ic50_recomputed"])
        for record in records
        if record["metadata"]["label"] == label
    ]


def fig_ic50_hist(records: list[dict], theme: dict[str, str], path: Path) -> None:
    """Show complete label separation across the IC50 threshold."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = [tick * 0.5 for tick in range(-12, 16)]
    ax.hist(
        [label_values(records, "Yes"), label_values(records, "No")],
        bins=bins,
        color=[theme["yes"], theme["no"]],
        edgecolor=theme["surface"],
        linewidth=1.5,
        label=["Yes (302)", "No (302)"],
        zorder=2,
    )
    recessive(ax, theme, axis="y")
    ax.set_xlim(-6.4, 7.4)
    log_ticks(ax)
    threshold_line(ax, theme)
    ax.set_xlabel("Recomputed IC50 (µM, log scale)")
    ax.set_ylabel("Questions")
    ax.set_title("IC50 cleanly separates Yes from No", color=theme["primary"])
    ax.legend(frameon=False, labelcolor=theme["secondary"], loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def cell_line_counts(
    records: list[dict],
) -> tuple[list[str], list[int], list[int]]:
    """Cell lines ordered by total coverage, with their Yes and No counts."""
    counts = collections.defaultdict(collections.Counter)
    for record in records:
        metadata = record["metadata"]
        counts[metadata["cell_line"]][metadata["label"]] += 1
    names = sorted(counts, key=lambda name: sum(counts[name].values()))
    return names, [counts[n]["Yes"] for n in names], [counts[n]["No"] for n in names]


def fig_cell_line_balance(
    records: list[dict], theme: dict[str, str], path: Path
) -> None:
    """Show outcome counts for each cell line, ordered by total coverage."""
    names, yes, no = cell_line_counts(records)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(names, yes, height=0.62, color=theme["yes"], label="Yes", zorder=2)
    ax.barh(
        names,
        no,
        left=yes,
        height=0.62,
        color=theme["no"],
        label="No",
        zorder=2,
    )
    recessive(ax, theme, axis="x")
    ax.set_xlim(0, max(y + n for y, n in zip(yes, no, strict=True)) * 1.13)
    ax.set_xlabel("Questions")
    ax.set_title("Every cell line has both outcomes", color=theme["primary"])
    ax.legend(frameon=False, labelcolor=theme["secondary"], loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def study_panel(ax: Axes, records: list[dict], theme: dict[str, str]) -> None:
    """Draw questions per source study, longest bar last, each one labelled."""
    studies = collections.Counter(record["metadata"]["study"] for record in records)
    names = [name for name, _ in studies.most_common()][::-1]
    values = [studies[name] for name in names]
    ax.barh(names, values, height=0.62, color=theme["yes"], zorder=2)
    for y, value in enumerate(values):
        ax.text(
            value + 8,
            y,
            str(value),
            va="center",
            color=theme["secondary"],
            fontsize=8.5,
        )
    ax.set_xlim(0, max(values) * 1.15)
    recessive(ax, theme, axis="x")
    ax.set_xlabel("Questions")
    ax.set_title("Questions per source study", color=theme["primary"], loc="left")


def drug_panel(ax: Axes, records: list[dict], theme: dict[str, str]) -> None:
    """Draw how many drugs are mentioned once, twice, and so on."""
    per_drug = collections.Counter(record["metadata"]["drug"] for record in records)
    frequency = collections.Counter(per_drug.values())
    counts = sorted(frequency)
    ax.bar(
        counts,
        [frequency[count] for count in counts],
        width=0.72,
        color=theme["yes"],
        zorder=2,
    )
    ax.set_yscale("log")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(count) for count in counts], rotation=45, ha="right")
    recessive(ax, theme, axis="y")
    ax.set_xlabel("Questions mentioning the drug")
    ax.set_ylabel("Distinct drugs (log scale)")
    ax.set_title("Most drugs appear once", color=theme["primary"], loc="left")


def fig_coverage(records: list[dict], theme: dict[str, str], path: Path) -> None:
    """Show source-study skew and sparse per-drug coverage, side by side."""
    fig, (study_ax, drug_ax) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    study_panel(study_ax, records, theme)
    drug_panel(drug_ax, records, theme)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Load the local sample and write each figure in light and dark themes."""
    records: list[dict] = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        style(theme)
        suffix = "" if mode == "light" else "-dark"
        fig_ic50_hist(records, theme, FIG_DIR / f"ic50-distribution{suffix}.png")
        fig_cell_line_balance(
            records, theme, FIG_DIR / f"ic50-by-cell-line{suffix}.png"
        )
        fig_coverage(records, theme, FIG_DIR / f"coverage{suffix}.png")
    print(f"Wrote 6 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
