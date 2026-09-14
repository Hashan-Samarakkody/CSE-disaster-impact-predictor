"""Render architecture/images/pipeline.png from the stage table below.

Kept as a script rather than a checked-in mystery PNG so the diagram can be
regenerated when a stage changes. Run: python scripts/make_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "architecture" / "images" / "pipeline.png"

INK, BLUE, ORANGE, GREEN, GREY = "#000000", "#0072B2", "#E69F00", "#009E73", "#8C8C8C"

# (stage label, what it produces, box colour)
STAGES = [
    ("01  Data acquisition", "market, disasters, macro, sp500", BLUE),
    ("02  Features & targets", "dataset, feature_spec, splits", BLUE),
    ("03  Exploratory analysis", "12 EDA figures, collinearity_rule", GREY),
    ("04  Regression", "results_regression, ablations", ORANGE),
    ("05  Classification", "results_classification, hurdle_table", ORANGE),
    ("06  Evaluation", "verdict_table, conformal_coverage", GREEN),
    ("07  Explainability", "SHAP figures", GREEN),
    ("08  Sector panel", "results_sector, sector figures", ORANGE),
    ("09  Synthesis", "written record (no code)", GREY),
]

SOURCES = ["data/  CSE archive (.xls)", "data/  EM-DAT export (.xlsx)",
           "World Bank / Yahoo (live)", "NASA POWER, DesInventar,\nFRED, Wikidata (live)"]


def _box(ax, x, y, w, h, text, colour, fontsize=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.4, edgecolor=colour, facecolor="white"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, fontweight="bold" if bold else "normal")


def _arrow(ax, xy_from, xy_to, colour=INK, style="-|>"):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, mutation_scale=11,
                                 linewidth=1.1, color=colour,
                                 shrinkA=2, shrinkB=2))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    ax.text(5, 11.3, "CSE disaster-impact pipeline", ha="center", fontsize=14,
            fontweight="bold", color=INK)
    ax.text(5, 11.0, "each stage reads and writes artifacts/; nothing is passed in memory",
            ha="center", fontsize=9, color=GREY)

    # Raw sources across the top.
    for i, s in enumerate(SOURCES):
        _box(ax, 0.25 + i * 2.45, 10.0, 2.2, 0.62, s, GREY, fontsize=7.5)
        _arrow(ax, (1.35 + i * 2.45, 10.0), (2.4, 9.32), GREY)

    # Stage boxes, one per row.
    y = 8.7
    centres = []
    for label, produces, colour in STAGES:
        _box(ax, 0.6, y, 3.6, 0.62, label, colour, fontsize=9.5, bold=True)
        _box(ax, 4.9, y, 4.5, 0.62, produces, GREY, fontsize=8)
        _arrow(ax, (4.2, y + 0.31), (4.9, y + 0.31), GREY)
        centres.append(y + 0.31)
        y -= 0.95

    # Sequential flow down the left edge.
    for a, b in zip(centres, centres[1:]):
        _arrow(ax, (2.4, a - 0.31), (2.4, b + 0.31), INK)

    # 06 and 07 consume the cached results of 04 and 05; 08 fits its own panel.
    for target in (centres[5], centres[6]):
        ax.add_patch(FancyArrowPatch((0.6, centres[3] - 0.31), (0.6, target + 0.31),
                                     connectionstyle="arc3,rad=0.42", arrowstyle="-|>",
                                     mutation_scale=9, linewidth=0.9, color=ORANGE,
                                     linestyle="--", shrinkA=2, shrinkB=2))

    ax.text(0.35, (centres[5] + centres[6]) / 2,
            "06 and 07 read the\ncached results of 04\nand 05; they refit\nnothing",
            ha="right", va="center", fontsize=7.5, color=ORANGE, style="italic")

    legend = ("blue = data construction     orange = model fitting     "
              "green = scoring     grey = descriptive / record")
    ax.text(5, 0.35, legend, ha="center", fontsize=8.5, color=GREY)
    ax.text(5, 0.05, "Figures land in docs/figures/.  Tables and fitted objects land in "
                     "artifacts/ (regenerable, not version-controlled).",
            ha="center", fontsize=8, color=GREY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
