"""Frozen closest-pair hard or confidence-aware soft NCMTL row sharing."""

import csv
import json
import os

import torch

from .candidate_row_distances import compute_candidate_row_distances


class RowTaskSharing:
    """Assign each candidate row to its closest pair and project that pair.

    Soft mode freezes one coefficient per row at the assignment epoch:
    beta = clip((relative_margin - soft_min) / (soft_full - soft_min), 0, 1).
    The selected rows move a beta fraction toward their pair centre after each
    optimizer update. Assignments and coefficients remain fixed thereafter.
    """

    PAIRS = (("ks_si", 0, 1), ("ks_er", 0, 2), ("si_er", 1, 2))

    def __init__(
        self,
        results_dir: str,
        task_names,
        warmup_mode: str = "fixed",
        warmup_min_epochs: int = 3,
        warmup_max_epochs: int = 10,
        warmup_stability_threshold: float = 0.90,
        warmup_stability_patience: int = 2,
        min_relative_margin: float = 0.0,
        sharing_mode: str = "hard",
        soft_min_margin: float = 0.0,
        soft_full_margin: float = 0.10,
    ):
        if list(task_names) != ["ks", "si", "er"]:
            raise ValueError("Row task sharing requires tasks ['ks', 'si', 'er']")
        self.task_names = list(task_names)
        self.warmup_mode = str(warmup_mode).strip().lower()
        if self.warmup_mode not in {"fixed", "adaptive"}:
            raise ValueError("row_warmup_mode must be 'fixed' or 'adaptive'")
        self.warmup_min_epochs = int(warmup_min_epochs)
        self.warmup_max_epochs = int(warmup_max_epochs)
        self.warmup_stability_threshold = float(warmup_stability_threshold)
        self.warmup_stability_patience = int(warmup_stability_patience)
        self.min_relative_margin = float(min_relative_margin)
        self.sharing_mode = str(sharing_mode).strip().lower()
        self.soft_min_margin = float(soft_min_margin)
        self.soft_full_margin = float(soft_full_margin)
        if self.warmup_min_epochs < 1:
            raise ValueError("row_warmup_min_epochs must be at least 1")
        if self.warmup_max_epochs < self.warmup_min_epochs:
            raise ValueError(
                "row_warmup_max_epochs must be >= row_warmup_min_epochs"
            )
        if not 0.0 <= self.warmup_stability_threshold <= 1.0:
            raise ValueError(
                "row_warmup_stability_threshold must satisfy 0 <= value <= 1"
            )
        if self.warmup_stability_patience < 1:
            raise ValueError("row_warmup_stability_patience must be at least 1")
        if not 0.0 <= self.min_relative_margin <= 1.0:
            raise ValueError("row_min_relative_margin must satisfy 0 <= value <= 1")
        if self.sharing_mode not in {"hard", "soft"}:
            raise ValueError("row_sharing_mode must be 'hard' or 'soft'")
        if not 0.0 <= self.soft_min_margin <= 1.0:
            raise ValueError("row_soft_min_margin must satisfy 0 <= value <= 1")
        if not 0.0 < self.soft_full_margin <= 1.0:
            raise ValueError("row_soft_full_margin must satisfy 0 < value <= 1")
        if self.soft_full_margin <= self.soft_min_margin:
            raise ValueError(
                "row_soft_full_margin must be greater than row_soft_min_margin"
            )

        self.assignments = None
        self.shared_row_mask = None
        self.sharing_coefficients = None
        self.assignment_epoch = None
        self.ready_to_freeze = False
        self.freeze_reason = None
        self._previous_proposed_assignments = None
        self._stable_transition_count = 0
        self.assignment_csv_path = os.path.join(
            results_dir, "row_pair_assignments.csv"
        )
        self.assignment_summary_path = os.path.join(
            results_dir, "row_pair_assignment_summary.json"
        )
        self.stability_path = os.path.join(
            results_dir, "row_pair_assignment_stability.csv"
        )
        self.warmup_stability_path = os.path.join(
            results_dir, "row_warmup_stability.csv"
        )
        self.warmup_summary_path = os.path.join(
            results_dir, "row_warmup_summary.json"
        )
        with open(self.stability_path, "w", newline="") as stability_file:
            csv.writer(stability_file).writerow(
                [
                    "epoch", "rows", "matching_frozen_assignment",
                    "match_rate", "observed_ks_si", "observed_ks_er",
                    "observed_si_er",
                ]
            )
        with open(self.warmup_stability_path, "w", newline="") as warmup_file:
            csv.writer(warmup_file).writerow(
                [
                    "epoch", "unchanged_rows", "stability_rate",
                    "stable_transition_count", "proposed_ks_si",
                    "proposed_ks_er", "proposed_si_er", "decision",
                ]
            )

    @property
    def initialized(self) -> bool:
        return self.assignments is not None

    @property
    def stable_transition_count(self) -> int:
        return self._stable_transition_count

    def initialize(self, weights, epoch: int) -> None:
        distances = compute_candidate_row_distances(weights, self.task_names)
        stacked = torch.stack([distances[name] for name, _, _ in self.PAIRS])
        sorted_distances, sorted_indices = torch.sort(stacked, dim=0)
        self.assignments = sorted_indices[0].to(dtype=torch.long)
        best_distances = sorted_distances[0]
        second_distances = sorted_distances[1]
        absolute_margins = second_distances - best_distances
        relative_margins = torch.where(
            second_distances > 0.0,
            absolute_margins / second_distances,
            torch.zeros_like(second_distances),
        )
        if self.sharing_mode == "soft":
            self.sharing_coefficients = torch.clamp(
                (relative_margins - self.soft_min_margin)
                / (self.soft_full_margin - self.soft_min_margin),
                min=0.0,
                max=1.0,
            )
            self.shared_row_mask = self.sharing_coefficients > 0.0
        else:
            self.shared_row_mask = relative_margins >= self.min_relative_margin
            self.sharing_coefficients = self.shared_row_mask.to(
                dtype=relative_margins.dtype
            )
        self.assignment_epoch = int(epoch)

        with open(self.assignment_csv_path, "w", newline="") as assignment_file:
            writer = csv.writer(assignment_file)
            writer.writerow(
                [
                    "row", "pair_id", "selected_pair", "selected_distance",
                    "second_distance", "absolute_margin", "relative_margin",
                    "sharing_coefficient", "sharing_decision",
                    "ks_si_distance", "ks_er_distance", "si_er_distance",
                ]
            )
            for row_index in range(stacked.shape[1]):
                selected = int(self.assignments[row_index].item())
                best = float(best_distances[row_index].item())
                second = float(second_distances[row_index].item())
                margin = float(absolute_margins[row_index].item())
                relative_margin = float(relative_margins[row_index].item())
                writer.writerow(
                    [
                        row_index,
                        selected,
                        self.PAIRS[selected][0],
                        best,
                        second,
                        margin,
                        relative_margin,
                        float(self.sharing_coefficients[row_index].item()),
                        self._sharing_decision(row_index),
                        float(stacked[0, row_index].item()),
                        float(stacked[1, row_index].item()),
                        float(stacked[2, row_index].item()),
                    ]
                )

        counts = self.assignment_counts()
        shared_counts = self.shared_assignment_counts()
        total = int(self.assignments.numel())
        shared_rows = int(torch.sum(self.shared_row_mask).item())
        fully_shared_rows = int(torch.sum(
            self.sharing_coefficients >= 1.0
        ).item())
        partially_shared_rows = int(torch.sum(
            (self.sharing_coefficients > 0.0)
            & (self.sharing_coefficients < 1.0)
        ).item())
        summary = {
            "strategy": "closest_pair",
            "sharing_mode": self.sharing_mode,
            "assignment_epoch": self.assignment_epoch,
            "frozen": True,
            "num_rows": total,
            "minimum_relative_margin": self.min_relative_margin,
            "soft_min_margin": self.soft_min_margin,
            "soft_full_margin": self.soft_full_margin,
            "pair_ids": {name: pair_id for pair_id, (name, _, _) in enumerate(self.PAIRS)},
            "counts": counts,
            "proportions": {name: count / total for name, count in counts.items()},
            "shared_pair_counts": shared_counts,
            "shared_rows": shared_rows,
            "partially_shared_rows": partially_shared_rows,
            "fully_shared_rows": fully_shared_rows,
            "independent_rows": total - shared_rows,
            "sharing_coverage": shared_rows / total,
            "sharing_coefficient_statistics": self.sharing_coefficient_statistics(),
        }
        with open(self.assignment_summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

    def _sharing_decision(self, row_index: int) -> str:
        coefficient = float(self.sharing_coefficients[row_index].item())
        if coefficient <= 0.0:
            return "independent"
        if self.sharing_mode == "hard":
            return "shared"
        if coefficient >= 1.0:
            return "fully_shared"
        return "partially_shared"

    def sharing_coefficient_statistics(self) -> dict[str, float]:
        if not self.initialized:
            return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        coefficients = self.sharing_coefficients.float()
        return {
            "mean": float(coefficients.mean().item()),
            "median": float(coefficients.median().item()),
            "std": float(coefficients.std(unbiased=False).item()),
            "min": float(coefficients.min().item()),
            "max": float(coefficients.max().item()),
        }

    def observe_adaptive_warmup(self, weights, epoch: int) -> bool:
        """Observe independent candidates and decide whether warm-up can stop."""
        if self.warmup_mode != "adaptive" or self.initialized:
            return False

        distances = compute_candidate_row_distances(weights, self.task_names)
        stacked = torch.stack([distances[name] for name, _, _ in self.PAIRS])
        proposed = torch.argmin(stacked, dim=0)
        total = int(proposed.numel())
        unchanged = None
        stability_rate = None
        if self._previous_proposed_assignments is not None:
            unchanged = int(torch.sum(
                proposed == self._previous_proposed_assignments
            ).item())
            stability_rate = unchanged / total

        eligible = epoch >= self.warmup_min_epochs
        if (
            eligible
            and stability_rate is not None
            and stability_rate >= self.warmup_stability_threshold
        ):
            self._stable_transition_count += 1
        elif eligible:
            self._stable_transition_count = 0

        decision = "continue"
        if self._stable_transition_count >= self.warmup_stability_patience:
            self.ready_to_freeze = True
            self.assignment_epoch = int(epoch)
            self.freeze_reason = "assignment stability reached"
            decision = "freeze_stable"
        elif epoch >= self.warmup_max_epochs:
            self.ready_to_freeze = True
            self.assignment_epoch = int(epoch)
            self.freeze_reason = "maximum adaptive warm-up reached"
            decision = "freeze_max_epoch"

        counts = {
            name: int(torch.sum(proposed == pair_id).item())
            for pair_id, (name, _, _) in enumerate(self.PAIRS)
        }
        with open(self.warmup_stability_path, "a", newline="") as warmup_file:
            csv.writer(warmup_file).writerow(
                [
                    epoch,
                    "" if unchanged is None else unchanged,
                    "" if stability_rate is None else stability_rate,
                    self._stable_transition_count,
                    counts["ks_si"], counts["ks_er"], counts["si_er"],
                    decision,
                ]
            )

        self._previous_proposed_assignments = proposed
        if self.ready_to_freeze:
            summary = {
                "mode": "adaptive",
                "minimum_epochs": self.warmup_min_epochs,
                "maximum_epochs": self.warmup_max_epochs,
                "stability_threshold": self.warmup_stability_threshold,
                "stability_patience": self.warmup_stability_patience,
                "actual_warmup_epochs": self.assignment_epoch,
                "freeze_reason": self.freeze_reason,
                "final_stability_rate": stability_rate,
                "final_stable_transition_count": self._stable_transition_count,
            }
            with open(self.warmup_summary_path, "w") as summary_file:
                json.dump(summary, summary_file, indent=2)
        return self.ready_to_freeze

    def assignment_counts(self) -> dict[str, int]:
        if not self.initialized:
            return {name: 0 for name, _, _ in self.PAIRS}
        return {
            name: int(torch.sum(self.assignments == pair_id).item())
            for pair_id, (name, _, _) in enumerate(self.PAIRS)
        }

    def shared_assignment_counts(self) -> dict[str, int]:
        if not self.initialized:
            return {name: 0 for name, _, _ in self.PAIRS}
        return {
            name: int(torch.sum(
                (self.assignments == pair_id) & self.shared_row_mask
            ).item())
            for pair_id, (name, _, _) in enumerate(self.PAIRS)
        }

    @torch.no_grad()
    def share(self, weights) -> None:
        if not self.initialized:
            return
        for pair_id, (_, first, second) in enumerate(self.PAIRS):
            cpu_row_mask = (
                (self.assignments == pair_id) & self.shared_row_mask
            )
            row_mask = cpu_row_mask.to(weights[first].device)
            if not bool(torch.any(row_mask)):
                continue
            row_indices = torch.nonzero(row_mask, as_tuple=False).squeeze(1)
            first_rows = weights[first][row_mask]
            second_rows = weights[second][row_mask]
            center = (first_rows + second_rows) / 2.0
            coefficients = self.sharing_coefficients[cpu_row_mask].to(
                device=weights[first].device,
                dtype=weights[first].dtype,
            ).unsqueeze(1)
            weights[first].index_copy_(
                0, row_indices, first_rows + coefficients * (center - first_rows)
            )
            weights[second].index_copy_(
                0, row_indices, second_rows + coefficients * (center - second_rows)
            )

    def cluster_loss(self, weights) -> torch.Tensor:
        if not self.initialized:
            return weights[0].new_zeros(())
        loss = weights[0].new_zeros(())
        for pair_id, (_, first, second) in enumerate(self.PAIRS):
            cpu_row_mask = (
                (self.assignments == pair_id) & self.shared_row_mask
            )
            row_mask = cpu_row_mask.to(weights[first].device)
            if not bool(torch.any(row_mask)):
                continue
            center = (
                (weights[first][row_mask] + weights[second][row_mask]) / 2.0
            ).detach()
            coefficients = self.sharing_coefficients[cpu_row_mask].to(
                device=weights[first].device,
                dtype=weights[first].dtype,
            ).unsqueeze(1)
            loss = loss + torch.sum(
                coefficients * (weights[first][row_mask] - center) ** 2
            )
            loss = loss + torch.sum(
                coefficients * (weights[second][row_mask] - center) ** 2
            )
        return loss

    def record_observed_stability(self, epoch: int, raw_values: dict) -> None:
        """Compare current closest pairs with the frozen warm-up assignments."""
        if not self.initialized:
            return
        stacked = torch.tensor(
            [raw_values[name] for name, _, _ in self.PAIRS], dtype=torch.float32
        )
        observed = torch.argmin(stacked, dim=0)
        matches = int(torch.sum(observed == self.assignments).item())
        counts = {
            name: int(torch.sum(observed == pair_id).item())
            for pair_id, (name, _, _) in enumerate(self.PAIRS)
        }
        total = int(observed.numel())
        with open(self.stability_path, "a", newline="") as stability_file:
            csv.writer(stability_file).writerow(
                [
                    epoch, total, matches, matches / total,
                    counts["ks_si"], counts["ks_er"], counts["si_er"],
                ]
            )
