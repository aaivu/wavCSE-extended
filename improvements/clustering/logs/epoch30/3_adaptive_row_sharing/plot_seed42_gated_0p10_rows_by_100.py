"""Show frozen seed-42, 10%-gated row assignments in 100-row blocks."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


CLUSTERING_ROOT = Path(__file__).resolve().parents[3]
SOURCE = (
    CLUSTERING_ROOT
    / "results_ncmtl_row_adaptive_gated_0p10_seed42_100ep"
    / "2026_09_21_15_01_16"
    / "row_pair_assignments.csv"
)
OUTPUT = Path(__file__).resolve().parent / "seed42_gated_0p10_pair_assignments_by_100_rows.png"
PAIRS = ("ks_si", "ks_er", "si_er")
LABELS = ("KS–SI", "KS–ER", "SI–ER", "Independent")
COLORS = ("#5B8FF9", "#F6BD16", "#5AD8A6", "#A4ACB3")
BLOCK_SIZE = 100


def read_rows():
    with SOURCE.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 2000 or [int(row["row"]) for row in rows] != list(range(2000)):
        raise ValueError("Expected exactly 2,000 consecutive row indices")
    if any(row["selected_pair"] not in PAIRS for row in rows):
        raise ValueError("Unexpected selected task pair")
    if any(row["sharing_decision"] not in {"shared", "independent"} for row in rows):
        raise ValueError("Unexpected sharing decision")
    return rows


def plot_stacked(ax, counts, labels, colors, title, block_labels):
    y = np.arange(len(block_labels))
    left = np.zeros(len(block_labels), dtype=int)
    for column, (label, color) in enumerate(zip(labels, colors)):
        widths = counts[:, column]
        ax.barh(y, widths, left=left, height=0.78, label=label, color=color)
        for index, width in enumerate(widths):
            if width >= 13:
                ax.text(left[index] + width / 2, index, str(width), ha="center",
                        va="center", fontsize=8, fontweight="bold", color="#17212B")
        left += widths
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, BLOCK_SIZE)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel("Rows in block (out of 100)")
    ax.set_yticks(y, block_labels)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)


def main():
    rows = read_rows()
    groups = [rows[start:start + BLOCK_SIZE] for start in range(0, 2000, BLOCK_SIZE)]
    block_labels = [f"{start:04d}–{start + 99:04d}"
                    for start in range(0, 2000, BLOCK_SIZE)]
    proposed = np.array([
        [sum(row["selected_pair"] == pair for row in group) for pair in PAIRS]
        for group in groups
    ])
    applied = np.array([
        [sum(row["selected_pair"] == pair and row["sharing_decision"] == "shared"
             for row in group) for pair in PAIRS]
        + [sum(row["sharing_decision"] == "independent" for row in group)]
        for group in groups
    ])
    if not np.all(proposed.sum(axis=1) == BLOCK_SIZE):
        raise ValueError("Proposed counts must sum to 100 in every block")
    if not np.all(applied.sum(axis=1) == BLOCK_SIZE):
        raise ValueError("Applied counts must sum to 100 in every block")

    fig, (ax_proposed, ax_applied) = plt.subplots(
        1, 2, figsize=(14, 11), sharey=True,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.12},
    )
    plot_stacked(ax_proposed, proposed, LABELS[:3], COLORS[:3],
                 "Closest pair for each row", block_labels)
    plot_stacked(ax_applied, applied, LABELS, COLORS,
                 "Actually shared after the 10% gate", block_labels)
    ax_proposed.invert_yaxis()
    ax_proposed.set_ylabel("Candidate-row index block")
    ax_applied.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle(
        "Seed 42 · 100-epoch run · frozen row decisions by 100-row block",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.legend(handles=[Patch(facecolor=color, label=label)
                        for label, color in zip(LABELS, COLORS)],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.045))
    fig.text(
        0.5, 0.015,
        "Assignments were frozen after epoch 4, then held fixed through epoch 100. "
        "Row indices are output-neuron positions, not spatial locations.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.92, bottom=0.11, left=0.10, right=0.98)
    fig.savefig(OUTPUT, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT}")
    print(f"Proposed totals: {proposed.sum(axis=0).tolist()}")
    print(f"Actually shared totals, then independent: {applied.sum(axis=0).tolist()}")


if __name__ == "__main__":
    main()
