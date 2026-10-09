"""Plot frozen adaptive all-row pair assignments for the three seed runs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "adaptive_all_row_assignment_heatmap.png"
STACKED_OUTPUT = Path(__file__).resolve().parent / "adaptive_all_pair_proportions.png"
STABILITY_OUTPUT = Path(__file__).resolve().parent / "adaptive_all_warmup_stability.png"
DISTANCE_OUTPUT = Path(__file__).resolve().parent / "adaptive_all_pair_distances_at_freeze.png"
SOURCES = {
    42: ROOT / "results_ncmtl/2026_09_21_10_14_28/row_pair_assignments.csv",
    43: ROOT / "results_ncmtl_row_adaptive_all_seed43/2026_09_21_12_22_05/row_pair_assignments.csv",
    44: ROOT / "results_ncmtl_row_adaptive_all_seed44/2026_09_21_12_26_49/row_pair_assignments.csv",
}
PAIRS = ("ks_si", "ks_er", "si_er")
LABELS = ("KS–SI", "KS–ER", "SI–ER")
COLORS = ("#5B8FF9", "#F6BD16", "#5AD8A6")


def read_assignments(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 2000 or [int(row["row"]) for row in rows] != list(range(2000)):
        raise ValueError(f"Expected rows 0–1999 exactly once in {path}")
    assignments = np.array([PAIRS.index(row["selected_pair"]) for row in rows])
    if not all(row["sharing_decision"] == "shared" for row in rows if "sharing_decision" in row):
        raise ValueError(f"Expected all-row sharing in {path}")
    return assignments


def read_warmup_stability(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        "epoch": [int(row["epoch"]) for row in rows if row["stability_rate"]],
        "rate": [float(row["stability_rate"]) for row in rows if row["stability_rate"]],
        "freeze_epoch": next(int(row["epoch"]) for row in rows
                             if row["decision"].startswith("freeze")),
    }


def read_pair_distances(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return [np.array([float(row[f"{pair}_distance"]) for row in rows])
            for pair in PAIRS]


def plot_proportions(seeds, counts):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    proportions = counts / counts.sum(axis=1, keepdims=True)
    left = np.zeros(len(seeds))
    for pair_id, (label, color) in enumerate(zip(LABELS, COLORS)):
        ax.barh(range(len(seeds)), proportions[:, pair_id], left=left,
                color=color, height=0.65, label=label)
        for seed_index, fraction in enumerate(proportions[:, pair_id]):
            ax.text(left[seed_index] + fraction / 2, seed_index,
                    f"{fraction:.1%}\n({counts[seed_index, pair_id]})",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#17212B")
        left += proportions[:, pair_id]
    ax.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Share of 2,000 candidate-row positions")
    ax.set_ylabel("Seed")
    ax.set_title("Frozen task-pair proportions after adaptive warm-up",
                 fontweight="bold")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False)
    fig.tight_layout()
    fig.savefig(STACKED_OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_warmup_stability(seeds):
    fig, ax = plt.subplots(figsize=(9, 5))
    line_colors = ("#3265A8", "#C57900", "#27866A")
    for seed, color in zip(seeds, line_colors):
        source = SOURCES[seed].with_name("row_warmup_stability.csv")
        values = read_warmup_stability(source)
        ax.plot(values["epoch"], np.array(values["rate"]) * 100,
                marker="o", linewidth=2.4, markersize=7,
                color=color, label=f"Seed {seed}")
    ax.axhline(90, color="#C0392B", linestyle="--", linewidth=1.5,
               label="90% threshold")
    ax.axvline(4, color="#555555", linestyle=":", linewidth=1.2)
    ax.set_xlim(1.9, 4.45)
    ax.set_ylim(80, 100)
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("Warm-up epoch")
    ax.set_ylabel("Rows retaining the previous epoch's proposed pair (%)")
    ax.set_title("Adaptive warm-up: row-pair assignment stability",
                 fontweight="bold")
    ax.text(3.96, 98.5, "Freeze after epoch 4", ha="right",
            color="#555555")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(STABILITY_OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_distances(seeds):
    fig, axes = plt.subplots(1, len(seeds), figsize=(13, 4.8), sharey=True)
    for ax, seed in zip(axes, seeds):
        distances = read_pair_distances(SOURCES[seed])
        parts = ax.violinplot(distances, positions=[1, 2, 3],
                              showmedians=True, showextrema=False)
        for body, color in zip(parts["bodies"], COLORS):
            body.set_facecolor(color)
            body.set_edgecolor("#263238")
            body.set_alpha(0.72)
        parts["cmedians"].set_color("#17212B")
        parts["cmedians"].set_linewidth(2)
        ax.set_xticks([1, 2, 3], LABELS)
        ax.set_title(f"Seed {seed}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Euclidean distance between corresponding weight rows")
    fig.suptitle("Task-pair row-distance distributions at freeze (epoch 4)",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.015,
             "Each violin contains 2,000 row distances; black line = median. "
             "Assignments use the closest pair separately for each row.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    fig.savefig(DISTANCE_OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    seeds = list(SOURCES)
    assignments = np.stack([read_assignments(SOURCES[seed]) for seed in seeds])
    counts = np.stack([np.bincount(row, minlength=3) for row in assignments])
    fig, (ax_heat, ax_counts) = plt.subplots(
        1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3.5, 1.25]}
    )
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax_heat.imshow(
        assignments, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto"
    )
    ax_heat.set_title("Frozen closest pair at each candidate-row position")
    ax_heat.set_xlabel("Candidate output row index (0–1999)")
    ax_heat.set_ylabel("Seed")
    ax_heat.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax_heat.set_xticks(range(0, 2001, 250))
    ax_heat.set_xlim(-0.5, 1999.5)
    ax_heat.tick_params(axis="y", length=0)

    left = np.zeros(len(seeds), dtype=int)
    for pair_id, (label, color) in enumerate(zip(LABELS, COLORS)):
        segment_centers = left + counts[:, pair_id] / 2
        ax_counts.barh(
            range(len(seeds)), counts[:, pair_id], left=left,
            color=color, label=label, height=0.65,
        )
        for seed_index, center in enumerate(segment_centers):
            ax_counts.text(
                center, seed_index, str(counts[seed_index, pair_id]),
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#17212B",
            )
        left += counts[:, pair_id]
    ax_counts.set_title("Assignment counts")
    ax_counts.set_xlabel("Rows")
    ax_counts.set_yticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax_counts.invert_yaxis()
    ax_counts.set_xlim(0, 2000)
    ax_counts.legend(loc="upper center", bbox_to_anchor=(0.5, -0.21), ncol=3,
                     frameon=False)

    fig.suptitle("Adaptive all-row NCMTL: task-pair assignments after epoch 4",
                 fontsize=14, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Each color is one frozen task pair; all 2,000 positions are shared. "
        "Row indices are output-neuron positions, not spatial locations.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(OUTPUT, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    plot_proportions(seeds, counts)
    plot_warmup_stability(seeds)
    plot_distances(seeds)
    print(f"Saved {OUTPUT}")
    print(f"Saved {STACKED_OUTPUT}")
    print(f"Saved {STABILITY_OUTPUT}")
    print(f"Saved {DISTANCE_OUTPUT}")
    print(f"Counts by seed: {dict(zip(seeds, counts.tolist()))}")


if __name__ == "__main__":
    main()
