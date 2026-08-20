"""Three-task Network Clustering for Multi-Task Learning model."""

from typing import List

import torch
import torch.nn as nn

from model.downstream_model import DownstreamMultiTaskModel, MultiClassifierOutput


class DownstreamMultiTaskModelNCMTL(DownstreamMultiTaskModel):
    """Baseline downstream network with one clusterable layer per task."""

    SUPPORTED_TASK_TYPE = "ks_si_er"

    def __init__(self, *args, **kwargs):
        task_type = kwargs.get("task_type")
        if task_type is None and len(args) >= 2:
            task_type = args[1]
        if task_type != self.SUPPORTED_TASK_TYPE:
            raise ValueError(
                "NCMTL v1 currently supports only task_type='ks_si_er' "
                "(three tasks; intent classification is not supported)."
            )
        super().__init__(*args, **kwargs)

        embedding_dim_shared2 = int(self.hidden_layer.out_features)
        output_dims = self._output_dims_from_task_type(task_type)
        self.candidate_dim = embedding_dim_shared2
        self.candidate_layers = nn.ModuleList(
            nn.Linear(embedding_dim_shared2, embedding_dim_shared2, bias=False)
            for _ in output_dims
        )
        self.classifiers = nn.ModuleList(
            nn.Linear(embedding_dim_shared2, output_dim) for output_dim in output_dims
        )

        self.register_buffer(
            "cluster_assignments",
            torch.full((len(output_dims),), -1, dtype=torch.long),
        )
        self.register_buffer("cluster_frozen", torch.tensor(False, dtype=torch.bool))

    def _shared_embedding(self, input_seq: torch.Tensor, apply_dropout: bool) -> torch.Tensor:
        embedding = self.projector_layer(input_seq)
        embedding = self.pooling.get_vector_after_pooling(embedding, dim=1)
        if apply_dropout:
            embedding = self.dropout_shared1(embedding)
        embedding = self.hidden_layer(embedding)
        if apply_dropout:
            embedding = self.dropout_shared2(embedding)
        return embedding

    def forward(self, input_seq):
        shared_embedding = self._shared_embedding(input_seq, apply_dropout=True)
        logits_list = []
        prediction_list = []
        for candidate, classifier in zip(self.candidate_layers, self.classifiers):
            logits = classifier(candidate(shared_embedding))
            logits_list.append(logits)
            prediction_list.append(torch.argmax(logits, dim=1))
        return MultiClassifierOutput(
            logits=tuple(logits_list), prediction=tuple(prediction_list)
        )

    def get_all_embeddings(self, input_seq):
        """Return the shared FC2 representation, before NCMTL candidates."""
        return self._shared_embedding(input_seq, apply_dropout=False)

    def get_candidate_weight_tensors(self) -> List[torch.Tensor]:
        return [layer.weight for layer in self.candidate_layers]

    def get_flattened_candidate_weights(self) -> torch.Tensor:
        return torch.stack(
            [weight.reshape(-1) for weight in self.get_candidate_weight_tensors()]
        )

    def has_valid_cluster_assignments(self) -> bool:
        return bool(torch.all(self.cluster_assignments >= 0).item())

    def set_cluster_assignments(self, assignments) -> None:
        assignments_tensor = torch.as_tensor(
            assignments, dtype=torch.long, device=self.cluster_assignments.device
        )
        if assignments_tensor.shape != self.cluster_assignments.shape:
            raise ValueError("cluster assignments must contain exactly three labels")
        self.cluster_assignments.copy_(assignments_tensor)

    def get_cluster_centers(self, detach: bool = False) -> dict[int, torch.Tensor]:
        if not self.has_valid_cluster_assignments():
            return {}
        centers = {}
        weights = self.get_candidate_weight_tensors()
        for cluster_id in torch.unique(self.cluster_assignments).tolist():
            members = [
                weights[index]
                for index, assignment in enumerate(self.cluster_assignments.tolist())
                if assignment == cluster_id
            ]
            center = torch.stack(members).mean(dim=0)
            centers[int(cluster_id)] = center.detach() if detach else center
        return centers

    def share_candidate_weights_by_cluster(self) -> None:
        centers = self.get_cluster_centers(detach=True)
        if not centers:
            return
        with torch.no_grad():
            for index, layer in enumerate(self.candidate_layers):
                cluster_id = int(self.cluster_assignments[index].item())
                layer.weight.copy_(centers[cluster_id])

    def get_cluster_loss(self) -> torch.Tensor:
        if not self.has_valid_cluster_assignments():
            return self.candidate_layers[0].weight.new_zeros(())
        centers = self.get_cluster_centers(detach=True)
        loss = self.candidate_layers[0].weight.new_zeros(())
        for index, layer in enumerate(self.candidate_layers):
            cluster_id = int(self.cluster_assignments[index].item())
            loss = loss + torch.sum((layer.weight - centers[cluster_id]) ** 2)
        return loss

    def get_cluster_state(self) -> dict:
        return {
            "assignments": self.cluster_assignments.detach().cpu().tolist(),
            "frozen": bool(self.cluster_frozen.item()),
        }
