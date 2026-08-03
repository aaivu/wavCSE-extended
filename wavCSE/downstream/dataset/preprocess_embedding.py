"""
Dataset loaders for downstream training using pre computed embeddings.

This module defines dataset wrappers that load pre extracted
embeddings from disk for downstream training and evaluation.
It supports multiple speech datasets and handles:
- Locating embeddings based on upstream model and pooling settings
- Selecting transformer layer subsets
- Mapping labels to task specific index tuples
- Combining datasets and creating percentage based subsets

These loaders are designed to be used by downstream multi task
training and evaluation pipelines.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import torch
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset, Subset

from torchaudio.datasets import SPEECHCOMMANDS, VoxCeleb1Identification, IEMOCAP, FluentSpeechCommands
from torchaudio.datasets.utils import _load_waveform

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Any

import os
import re

from utils.pooling_id import make_pooling_id

# Speech Commands dataset pre-processing
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.01"
class SPEECHCOMMANDSEmbedding(SPEECHCOMMANDS):
    def __init__(
        self,
        root: Union[str, Path],
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        frame_pooling_param: Optional[Union[int, float]],
        upstream_model_type: str,
        label_mapping: Dict[str, int],
        transformer_layer_array: Optional[List[int]] = None,
        index_pattern: Optional[str] = None,
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download, subset)

        self.root_embedding = str(root_embedding)
        self.frame_pooling_type = frame_pooling_type
        self.frame_pooling_param = frame_pooling_param
        self.frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
        self.upstream_model_type = upstream_model_type
        self.label_mapping = label_mapping
        self.transformer_layer_array = transformer_layer_array

        if index_pattern is None:
            index_pattern = "1"
        if not index_pattern or any(c not in ("0", "1") for c in index_pattern):
            raise ValueError("index_pattern must be None or a string of '0' and '1'.")

        self.index_pattern = index_pattern
        self.device = device or torch.device("cpu")

    def _build_index_tuple(self, label_index: int) -> Tuple[int, ...]:
        return tuple(label_index if c == "1" else -1 for c in self.index_pattern)

    def get_metadata(self, n: int):
        metadata = super().get_metadata(n)

        wav_relpath = metadata[0]
        label = str(metadata[2])

        emb_relpath = wav_relpath.replace(
            ".wav", f"_{self.upstream_model_type}_{self.frame_pool_id}.pt"
        )
        emb_path = os.path.join(self.root_embedding, emb_relpath)

        try:
            embedding = torch.load(emb_path, map_location=self.device)
        except (FileNotFoundError, RuntimeError, EOFError) as e:
            raise RuntimeError(f"Failed to load embedding: {emb_path}") from e

        if self.transformer_layer_array is not None:
            embedding = embedding[self.transformer_layer_array, :]

        label_index = self.label_mapping.get(label, 10)
        index_tuple = self._build_index_tuple(label_index)

        return embedding, torch.tensor(index_tuple, dtype=torch.long)

    def __getitem__(self, n: int):
        return self.get_metadata(n)

    
# Voxceleb dataset pre-processing
_IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
class VoxCeleb1Embedding(VoxCeleb1Identification):
    def __init__(
        self,
        root: Union[str, Path],
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        frame_pooling_param: Optional[Union[int, float]],
        upstream_model_type: str,
        label_mapping: Dict[str, int],
        subset: str = "train",
        transformer_layer_array: Optional[List[int]] = None,
        index_pattern: Optional[str] = None,
        download: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(root, subset, _IDEN_SPLIT_URL, download)

        self.root_embedding = str(root_embedding)
        self.frame_pooling_type = frame_pooling_type
        self.frame_pooling_param = frame_pooling_param
        self.frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
        self.upstream_model_type = upstream_model_type
        self.label_mapping = label_mapping
        self.transformer_layer_array = transformer_layer_array

        if index_pattern is None:
            index_pattern = "1"
        if not index_pattern or any(c not in ("0", "1") for c in index_pattern):
            raise ValueError("index_pattern must be None or a string containing only '0' and '1'.")
        self.index_pattern = index_pattern

        self.device = device or torch.device("cpu")

    def _build_index_tuple(self, label_index: int) -> Tuple[int, ...]:
        return tuple(label_index if c == "1" else -1 for c in self.index_pattern)

    def get_metadata(self, n: int):
        metadata = super().get_metadata(n)

        wav_relpath = metadata[0]  # relative wav path under VoxCeleb root
        label = str(metadata[2])   # speaker id string

        emb_relpath = wav_relpath.replace(
            ".wav", f"_{self.upstream_model_type}_{self.frame_pool_id}.pt"
        )
        emb_path = os.path.join(self.root_embedding, emb_relpath)

        try:
            embedding = torch.load(emb_path, map_location=self.device)
        except (FileNotFoundError, RuntimeError, EOFError) as e:
            raise RuntimeError(f"Failed to load embedding: {emb_path}") from e

        if self.transformer_layer_array is not None:
            embedding = embedding[self.transformer_layer_array, :]

        # Speaker label index
        label_index = self.label_mapping.get(label, -1)
        index_tuple = self._build_index_tuple(label_index)

        return embedding, torch.tensor(index_tuple, dtype=torch.long)

    def __getitem__(self, n: int):
        return self.get_metadata(n)


# IEMOCAP dataset pre-processing
class IEMOCAPEmbedding(IEMOCAP):
    def __init__(
        self,
        root: Union[str, Path],
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        frame_pooling_param: Optional[Union[int, float]],
        upstream_model_type: str,
        sessions: List[str],
        transformer_layer_array: Optional[List[int]] = None,
        utterance_type: Optional[str] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        index_pattern: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        sessions_tuple, speakers_tuple = self.process_sessions(sessions)

        super().__init__(root, sessions_tuple, utterance_type)

        self.root_embedding = str(root_embedding)
        self.frame_pooling_type = frame_pooling_type
        self.frame_pooling_param = frame_pooling_param
        self.frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
        self.upstream_model_type = upstream_model_type
        self.transformer_layer_array = transformer_layer_array
        self.label_mapping = label_mapping or {}

        # Filter out "fru" and keep only requested speakers
        self.data = [wav_stem for wav_stem in self.data if self.mapping[wav_stem]["label"] != "fru"]
        self.data = [
            wav_stem for wav_stem in self.data
            if self.mapping[wav_stem]["path"].split("/")[3].split("_")[0] in speakers_tuple
        ]

        if index_pattern is None:
            index_pattern = "1"
        if not index_pattern or any(c not in ("0", "1") for c in index_pattern):
            raise ValueError("index_pattern must be None or a string containing only '0' and '1'.")
        self.index_pattern = index_pattern

        self.device = device or torch.device("cpu")

    @staticmethod
    def process_sessions(input_array: List[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        session_id_array: List[str] = []
        speaker_id_array: List[str] = []

        for element in input_array:
            match = re.search(r"\d+", str(element))
            numeric_part = match.group() if match else ""
            session_id_array.append(numeric_part)

            processed_output = "Ses0" + str(element)

            # If element is already a speaker id like "Ses01F", numeric_part != element
            if numeric_part != str(element):
                speaker_id_array.append(processed_output)
            else:
                speaker_id_array.extend([processed_output + "F", processed_output + "M"])

        return tuple(session_id_array), tuple(speaker_id_array)

    def _build_index_tuple(self, label_index: int) -> Tuple[int, ...]:
        return tuple(label_index if c == "1" else -1 for c in self.index_pattern)

    def get_metadata(self, n: int):
        metadata = super().get_metadata(n)

        wav_relpath = metadata[0]  # relative wav path
        label = str(metadata[3])   # emotion label string

        emb_relpath = wav_relpath.replace(
            ".wav", f"_{self.upstream_model_type}_{self.frame_pool_id}.pt"
        )
        emb_path = os.path.join(self.root_embedding, emb_relpath)

        try:
            embedding = torch.load(emb_path, map_location=self.device)
        except (FileNotFoundError, RuntimeError, EOFError) as e:
            raise RuntimeError(f"Failed to load embedding: {emb_path}") from e

        if self.transformer_layer_array is not None:
            embedding = embedding[self.transformer_layer_array, :]

        # Map label -> index (treat "exc" as "hap" like your original)
        if label == "exc":
            label_index = self.label_mapping.get("hap", -1)
        else:
            label_index = self.label_mapping.get(label, -1)

        index_tuple = self._build_index_tuple(label_index)
        return embedding, torch.tensor(index_tuple, dtype=torch.long)

    def __getitem__(self, n: int):
        return self.get_metadata(n)

    
class FluentSpeechCommandEmbedding(FluentSpeechCommands):
    def __init__(
        self,
        root: Union[str, Path],
        root_embedding: Union[str, Path],
        frame_pooling_type: str,
        frame_pooling_param: Optional[Union[int, float]],
        upstream_model_type: str,
        label_mapping: Dict[str, int],
        subset: str = "train",
        transformer_layer_array: Optional[List[int]] = None,
        index_pattern: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(root=str(root), subset=subset)

        self.root_embedding = str(root_embedding)
        self.frame_pooling_type = frame_pooling_type
        self.frame_pooling_param = frame_pooling_param
        self.frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
        self.upstream_model_type = upstream_model_type
        self.label_mapping = label_mapping
        self.transformer_layer_array = transformer_layer_array

        # Default behavior: single index (same as earlier code)
        if index_pattern is None:
            index_pattern = "1"
        if not index_pattern or any(c not in ("0", "1") for c in index_pattern):
            raise ValueError("index_pattern must be None or contain only '0' and '1'")

        self.index_pattern = index_pattern
        self.device = device or torch.device("cpu")

    def _build_index_tuple(self, label_index: int) -> Tuple[int, ...]:
        return tuple(label_index if c == "1" else -1 for c in self.index_pattern)

    @staticmethod
    def _normalize(text: str) -> str:
        return str(text).lower().strip().replace(" ", "_")

    def get_metadata(self, n: int):
        metadata = super().get_metadata(n)

        # Relative wav path (used ONLY to locate embedding)
        wav_relpath = metadata[0]

        # Build canonical label (same as saving script)
        action = self._normalize(metadata[5])
        object_ = self._normalize(metadata[6])
        location = self._normalize(metadata[7])

        parts = [action, object_, location]
        parts = [p for p in parts if p != "none"]
        label = "_".join(parts)

        label_index = self.label_mapping.get(label, -1)

        # Load embedding
        emb_relpath = wav_relpath.replace(
            ".wav", f"_{self.upstream_model_type}_{self.frame_pool_id}.pt"
        )
        emb_path = os.path.join(self.root_embedding, emb_relpath)

        try:
            embedding = torch.load(emb_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding: {emb_path}") from e

        if self.transformer_layer_array is not None:
            embedding = embedding[self.transformer_layer_array, :]

        index_tuple = self._build_index_tuple(label_index)

        return embedding, torch.tensor(index_tuple, dtype=torch.long)

    def __getitem__(self, n: int):
        return self.get_metadata(n)
    

# Combining multiple datasets
class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.total_length = sum(len(dataset) for dataset in datasets)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)

# Taking subset of data        
class PercentageSubset(Subset):
    def __init__(self, dataset: Dataset, percentage: int) -> None:
        if not (0 <= percentage <= 100):
            raise ValueError("Percentage must be between 0 and 100")

        total_samples = len(dataset)

        if percentage == 0:
            indices = []
            super().__init__(dataset, indices)
            return

        num_samples = max(1, total_samples * percentage // 100)
        num_samples = min(num_samples, total_samples)

        # evenly spaced indices across the dataset
        step = total_samples / num_samples
        indices = [int(i * step) for i in range(num_samples)]

        super().__init__(dataset, indices)

