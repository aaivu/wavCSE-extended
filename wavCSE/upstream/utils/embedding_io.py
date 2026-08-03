"""
Utilities for writing extracted embeddings and metadata to disk.

This file handles:
1. Saving per utterance embeddings as .pt files
2. Creating a structured directory hierarchy based on
   upstream model type and pooling configuration
3. Writing a CSV file that tracks wav paths, labels,
   and embedding file sizes
4. Supporting incremental writes across dataset splits
   (train, validation, test)

This utility is used by the upstream embedding extraction
orchestration script.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import logging
import torch
from utils.get_size import get_file_size
from utils.pooling_id import make_pooling_id

def process_and_write_data(
    data,
    root_emb_path,
    upstream_model_type,
    frame_pooling_type,
    dataset_name,
    limit: int = None,
    append: bool = False,
    frame_pooling_param=None,   # NEW: pass pooling param here
):
    root_emb_path = os.path.abspath(os.path.expanduser(root_emb_path))

    frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)  

    dir_emb_path = os.path.join(root_emb_path, upstream_model_type, frame_pool_id, dataset_name)
    os.makedirs(dir_emb_path, exist_ok=True)

    csv_path = os.path.join(
        dir_emb_path,
        f"{dataset_name}_{upstream_model_type}_{frame_pool_id}.csv"
    )

    logging.info("Path embedding: %s", dir_emb_path)

    mode = "a" if append else "w"
    write_header = (not append) or (not os.path.exists(csv_path))

    with open(csv_path, mode) as csv_file:
        if write_header:
            csv_file.write("wavpath,label,file_size\n")

        for index, (embedding, label, wavpath) in enumerate(data):
            if limit is not None and index >= limit:
                break

            embpath = os.path.join(dir_emb_path, wavpath)
            embpath = embpath.replace(".wav", f"_{upstream_model_type}_{frame_pool_id}.pt")

            os.makedirs(os.path.dirname(embpath), exist_ok=True)
            torch.save(embedding, embpath)

            csv_file.write(f"{wavpath},{label},{get_file_size(embpath)}\n")

            if (index + 1) % 200 == 0:
                logging.info("[%s] Processed %d files", dataset_name, index + 1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logging.info("CSV written: %s", csv_path)
