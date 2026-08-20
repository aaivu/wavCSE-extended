"""K-Means++ helpers for the three NCMTL candidate networks."""

import logging
import warnings
from typing import Iterable, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning


def canonicalize_cluster_labels(labels: Iterable[int]) -> list[int]:
    """Remap arbitrary cluster IDs by their smallest task-member index."""
    labels = [int(label) for label in labels]
    members = {}
    for task_index, label in enumerate(labels):
        members.setdefault(label, []).append(task_index)
    ordered_labels = sorted(members, key=lambda label: min(members[label]))
    remapping = {old: new for new, old in enumerate(ordered_labels)}
    return [remapping[label] for label in labels]


def cluster_candidate_weights(
    task_parameter_matrix: torch.Tensor,
    num_clusters: int,
    random_state: int,
    n_init: int = 1,
    max_iter: int = 100,
    previous_assignments: Optional[Iterable[int]] = None,
) -> list[int]:
    """Run CPU K-Means++ and return deterministic canonical assignments."""
    if task_parameter_matrix.ndim != 2 or task_parameter_matrix.shape[0] != 3:
        raise ValueError("NCMTL expects a task parameter matrix with shape [3, P]")
    if not 1 <= int(num_clusters) <= 3:
        raise ValueError("num_clusters must satisfy 1 <= K <= 3")

    matrix = task_parameter_matrix.detach().cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        labels = KMeans(
            n_clusters=int(num_clusters),
            init="k-means++",
            n_init=int(n_init),
            max_iter=int(max_iter),
            random_state=int(random_state),
        ).fit_predict(matrix)

    assignments = canonicalize_cluster_labels(labels.tolist())
    if len(np.unique(assignments)) == int(num_clusters):
        return assignments

    if previous_assignments is not None:
        previous = canonicalize_cluster_labels(previous_assignments)
        if len(np.unique(previous)) == int(num_clusters):
            logging.warning(
                "K-Means++ found fewer than %d non-empty clusters; retaining the "
                "previous valid NCMTL assignment.",
                num_clusters,
            )
            return previous

    raise RuntimeError(
        f"K-Means++ could not form {num_clusters} distinct task parameter groups; "
        "the candidate weight vectors may have collapsed to identical points."
    )
