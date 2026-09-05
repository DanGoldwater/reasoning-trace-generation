"""Generate the EDA figures for docs/eda_report.md.

Run with:  uv run --with matplotlib python docs/make_plots.py
Figures are written to docs/figures/ in light and dark variants.
"""

import collections
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_PATH = Path("data/private_qa.json")
FIG_DIR = Path("docs/figures")
THRESHOLD = 0.1  # micromolar: the apparent "sensitive" cut-off

# Values taken unchanged from the dataviz reference palette.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
    },
}


def style(t: dict[str, str]) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": t["primary"],
            "axes.labelcolor": t["secondary"],
            "axes.edgecolor": t["axis"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "grid.color": t["grid"],
            "grid.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "font.size": 9,
        }
    )


def recessive(ax, t: dict[str, str], axis: str = "x") -> None:
    ax.grid(axis=axis, linestyle="-", linewidth=0.8, color=t["grid"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def threshold_line(ax, t: dict[str, str]) -> None:
    """Dashed marker for the 0.1 uM cut-off: a real threshold, not a gridline."""
    x = math.log10(THRESHOLD)
    ax.axvline(x, color=t["secondary"], lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(
        x + 0.09,
        ax.get_ylim()[1],
        "0.1 µM cut-off",
        color=t["secondary"],
        rotation=90,
        ha="left",
        va="top",
        fontsize=8.5,
    )


def fig_ic50_hist(records, t, path) -> None:
    logs = [math.log10(r["metadata"]["ic50_recomputed"]) for r in records]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = [x * 0.5 for x in range(-12, 0)]
    ax.hist(
        logs,
        bins=bins,
        color=t["series"],
        edgecolor=t["surface"],
        linewidth=2,
        zorder=2,
    )
    recessive(ax, t, axis="y")
    ax.set_xlim(-6.4, -0.62)
    threshold_line(ax, t)
    ax.set_xticks(list(range(-6, 0)))
    ax.set_xticklabels([f"$10^{{{v}}}$" for v in range(-6, 0)])
    ax.set_xlabel("Recomputed IC50 (µM, log scale)")
    ax.set_ylabel("Questions")
    ax.set_title(
        "Every IC50 sits below 0.1 µM, most of them just below", color=t["primary"]
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_ic50_by_cell_line(records, t, path) -> None:
    by = collections.defaultdict(list)
    for r in records:
        by[r["metadata"]["cell_line"]].append(
            math.log10(r["metadata"]["ic50_recomputed"])
        )
    order = sorted(by, key=lambda k: sum(by[k]) / len(by[k]))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for i, name in enumerate(order):
        ax.scatter(
            by[name],
            [i] * len(by[name]),
            s=52,
            color=t["series"],
            edgecolor=t["surface"],
            linewidth=2,
            zorder=3,
            clip_on=False,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.7, len(order) - 0.3)
    recessive(ax, t, axis="x")
    ax.set_xlim(-6.4, -0.62)
    threshold_line(ax, t)
    ax.set_xticks(list(range(-6, 0)))
    ax.set_xticklabels([f"$10^{{{v}}}$" for v in range(-6, 0)])
    ax.set_xlabel("Recomputed IC50 (µM, log scale)")
    ax.set_title("IC50 spread per cell line (ordered by mean)", color=t["primary"])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_coverage(records, t, path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))

    studies = collections.Counter(r["metadata"]["study"] for r in records)
    names = [k for k, _ in studies.most_common()][::-1]
    vals = [studies[k] for k in names]
    ax1.barh(names, vals, height=0.62, color=t["series"], zorder=2)
    for y, v in enumerate(vals):
        ax1.text(v + 1, y, str(v), va="center", color=t["secondary"], fontsize=8.5)
    ax1.set_xlim(0, max(vals) * 1.18)
    recessive(ax1, t, axis="x")
    ax1.set_xlabel("Questions")
    ax1.set_title("Questions per source study", color=t["primary"], loc="left")

    per_drug = collections.Counter(r["metadata"]["drug"] for r in records)
    hist = collections.Counter(per_drug.values())
    ks = sorted(hist)
    ax2.bar(
        [str(k) for k in ks],
        [hist[k] for k in ks],
        width=0.62,
        color=t["series"],
        zorder=2,
    )
    for x, k in enumerate(ks):
        ax2.text(
            x,
            hist[k] + 0.6,
            str(hist[k]),
            ha="center",
            color=t["secondary"],
            fontsize=8.5,
        )
    ax2.set_ylim(0, max(hist.values()) * 1.2)
    recessive(ax2, t, axis="y")
    ax2.set_xlabel("Questions mentioning the drug")
    ax2.set_ylabel("Distinct drugs")
    ax2.set_title("Drugs, by how often they appear", color=t["primary"], loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    records = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        style(theme)
        suffix = "" if mode == "light" else "-dark"
        fig_ic50_hist(records, theme, FIG_DIR / f"ic50-distribution{suffix}.png")
        fig_ic50_by_cell_line(
            records, theme, FIG_DIR / f"ic50-by-cell-line{suffix}.png"
        )
        fig_coverage(records, theme, FIG_DIR / f"coverage{suffix}.png")
    print(f"Wrote 6 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
