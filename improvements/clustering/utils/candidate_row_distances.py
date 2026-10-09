"""Read-only candidate-row distance diagnostics for NCMTL runs."""

import csv
import json
import os
from itertools import combinations
from typing import Sequence

import torch


@torch.no_grad()
def compute_candidate_row_distances(
    weights: Sequence[torch.Tensor],
    task_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Return corresponding-row L2 distances for every task pair on CPU."""
    if len(weights) != len(task_names):
        raise ValueError("weights and task_names must have the same length")

    distances = {}
    for first_index, second_index in combinations(range(len(weights)), 2):
        pair_name = f"{task_names[first_index]}_{task_names[second_index]}"
        distances[pair_name] = torch.linalg.vector_norm(
            weights[first_index].detach() - weights[second_index].detach(),
            dim=1,
        ).cpu()
    return distances


class CandidateRowDistanceLogger:
    """Collect and persist one pre-sharing row-distance snapshot per epoch."""

    def __init__(self, results_dir: str, task_names: Sequence[str]):
        self.task_names = list(task_names)
        if self.task_names != ["ks", "si", "er"]:
            raise ValueError(
                "Candidate row-distance matrices currently require tasks "
                "['ks', 'si', 'er']"
            )

        self.summary_path = os.path.join(
            results_dir, "candidate_row_distance_summary.csv"
        )
        self.matrices_path = os.path.join(
            results_dir, "candidate_row_distance_matrices.json"
        )
        self.values_path = os.path.join(
            results_dir, "candidate_row_distance_values.json"
        )
        self._matrices = []
        self._values = []

        with open(self.summary_path, "w", newline="") as summary_file:
            csv.writer(summary_file).writerow(
                [
                    "epoch", "batch", "stage", "pair", "mean", "std", "min",
                    "q25", "median", "q75", "max",
                ]
            )
        for path in (self.matrices_path, self.values_path):
            with open(path, "w") as output_file:
                json.dump([], output_file)

    def capture(self, model, epoch: int, batch: int) -> dict:
        raw_distances = compute_candidate_row_distances(
            model.get_candidate_weight_tensors(), self.task_names
        )
        summaries = {}
        for pair_name, values in raw_distances.items():
            summaries[pair_name] = {
                "mean": float(values.mean().item()),
                "std": float(values.std().item()),
                "min": float(values.min().item()),
                "q25": float(torch.quantile(values, 0.25).item()),
                "median": float(torch.quantile(values, 0.50).item()),
                "q75": float(torch.quantile(values, 0.75).item()),
                "max": float(values.max().item()),
            }

        return {
            "epoch": epoch,
            "batch": batch,
            "stage": "post_optimizer_pre_sharing",
            "assignments_before_sharing": (
                model.cluster_assignments.detach().cpu().tolist()
                if model.has_valid_cluster_assignments()
                else None
            ),
            "clusters_frozen": bool(model.cluster_frozen.item()),
            "summaries": summaries,
            "values": {
                pair_name: values.tolist()
                for pair_name, values in raw_distances.items()
            },
        }

    def write(self, snapshot: dict) -> None:
        with open(self.summary_path, "a", newline="") as summary_file:
            writer = csv.writer(summary_file)
            for pair_name, values in snapshot["summaries"].items():
                writer.writerow(
                    [
                        snapshot["epoch"], snapshot["batch"], snapshot["stage"],
                        pair_name, values["mean"], values["std"], values["min"],
                        values["q25"], values["median"], values["q75"], values["max"],
                    ]
                )

        means = snapshot["summaries"]
        self._matrices.append(
            {
                "epoch": snapshot["epoch"],
                "batch": snapshot["batch"],
                "stage": snapshot["stage"],
                "tasks": self.task_names,
                "assignments_before_sharing": snapshot["assignments_before_sharing"],
                "clusters_frozen": snapshot["clusters_frozen"],
                "matrix": [
                    [0.0, means["ks_si"]["mean"], means["ks_er"]["mean"]],
                    [means["ks_si"]["mean"], 0.0, means["si_er"]["mean"]],
                    [means["ks_er"]["mean"], means["si_er"]["mean"], 0.0],
                ],
            }
        )
        with open(self.matrices_path, "w") as matrices_file:
            json.dump(self._matrices, matrices_file, indent=2)

        self._values.append(
            {
                key: snapshot[key]
                for key in (
                    "epoch", "batch", "stage", "assignments_before_sharing",
                    "clusters_frozen", "values",
                )
            }
        )
        with open(self.values_path, "w") as values_file:
            json.dump(self._values, values_file)
