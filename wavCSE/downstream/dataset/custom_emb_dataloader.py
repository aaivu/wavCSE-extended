"""
Custom DataLoader for embedding based datasets.

This module defines a lightweight wrapper around PyTorch's
DataLoader for handling pre computed embedding datasets.
It provides a custom collate function that stacks embedding
tensors and corresponding label tensors into batched inputs
suitable for downstream training.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import torch
from torch.utils.data import DataLoader

class CustomEmbDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
        persistent_workers=False
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.defined_collate,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers
        )

    def defined_collate(self, batch):
        # Expect each item as: (embedding, label_tensor)
        sequences, labels = zip(*batch)
        input_seq = torch.stack(sequences)
        labels = torch.stack(labels).long()
        return input_seq, labels
