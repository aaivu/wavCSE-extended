"""
Dataset specific embedding extraction modules.

This module defines dataset wrappers for extracting embeddings
from multiple speech datasets using an upstream SSL model.
It includes dataset specific preprocessing, silence aware
trimming, optional resampling, label mapping, and temporal
pooling to produce fixed dimensional representations.

Supported datasets include:
- Speech Commands
- VoxCeleb1
- IEMOCAP
- Fluent Speech Commands

These classes are designed to integrate seamlessly with
the upstream embedding extraction pipeline.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import torch
from torch import Tensor
from torch.hub import download_url_to_file
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F

from torchaudio.datasets import SPEECHCOMMANDS, VoxCeleb1Identification, IEMOCAP, FluentSpeechCommands
from torchaudio.datasets.utils import _load_waveform
import torchaudio
import torchaudio.transforms as T

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List, Any
from scipy import stats

from pooling.pooling import *

import os
import re
import numpy as np

MAX_WAV_BYTES = int(1.7 * 1024 * 1024)   # 1.7 MB
MAX_SECONDS = 60.0

def _safe_get_filesize_bytes(full_path: str) -> Optional[int]:
    try:
        return os.path.getsize(full_path)
    except Exception:
        return None

def trim_to_non_silent_window(
    waveform: Tensor,
    sample_rate: int,
    target_seconds: float = MAX_SECONDS,
) -> Tensor:
    """
    Pick the target_seconds-long window with maximum short-time energy.
    This tends to avoid silence because silence has low energy.

    waveform: (C, T) or (T,)
    returns: same shape, but T trimmed to target_seconds
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)

    target_len = int(sample_rate * target_seconds)
    if waveform.size(-1) <= target_len:
        return waveform

    # Energy computed on mono for stability; we still return original channels sliced
    mono = waveform.mean(dim=0)  # (T,)

    frame_len = max(1, int(0.025 * sample_rate))  # 25 ms
    hop = max(1, int(0.010 * sample_rate))        # 10 ms

    if mono.numel() < frame_len:
        return waveform[..., :target_len]

    frames = mono.unfold(0, frame_len, hop)  # (N, frame_len)
    energy = (frames * frames).mean(dim=1)   # (N,)

    # How many frames correspond to target_len?
    win_frames = max(1, (target_len - frame_len) // hop + 1)

    # Moving sum of energy over win_frames
    e = energy.view(1, 1, -1)
    kernel = torch.ones(1, 1, win_frames, device=energy.device, dtype=energy.dtype)
    sums = F.conv1d(e, kernel, stride=1)  # (1,1,N-win_frames+1)

    best = int(sums.argmax().item())
    start = best * hop
    end = start + target_len
    return waveform[..., start:end]

def maybe_trim_by_filesize(
    waveform: Tensor,
    sample_rate: int,
    file_size_bytes: Optional[int],
    max_bytes: int = MAX_WAV_BYTES,
    target_seconds: float = MAX_SECONDS,
) -> Tensor:
    """
    If file_size_bytes > max_bytes, trim to target_seconds using energy-based windowing.
    If file_size_bytes is unavailable (None), we fall back to duration check.
    """
    target_len = int(sample_rate * target_seconds)

    if file_size_bytes is not None:
        if file_size_bytes <= max_bytes:
            return waveform
        # filesize triggers trimming, but still only if longer than target
        if waveform.size(-1) <= target_len:
            return waveform
        return trim_to_non_silent_window(waveform, sample_rate, target_seconds)

    # Fallback when we cannot read filesize reliably
    if waveform.size(-1) <= target_len:
        return waveform
    return trim_to_non_silent_window(waveform, sample_rate, target_seconds)

# Speech Commands dataset pre-processing
FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.01"
class SPEECHCOMMANDSEmbedding(SPEECHCOMMANDS):
    def __init__(
        self,
        root: Union[str, Path],
        frame_pooling_type: str = None,
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
        model: torch.nn.Module = None,
        device: torch.device = None,
        label_mapping: Dict[str, int] = None,
        frame_pooling_param: Optional[int] = None,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download, subset)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.label_mapping = label_mapping
        self.device = device
        
        if self.device is None:
            self.device = torch.device("cpu")
        
        self.frame_pooling = Pooling(frame_pooling_type, frame_pooling_param)
        self.model = self.model.to(self.device)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)
        waveform = _load_waveform(self._archive, metadata[0], metadata[1])
        sample_rate = int(metadata[1])

        full_path = None
        if hasattr(self, "_path"):
            full_path = os.path.join(self._path, metadata[0])
        file_bytes = _safe_get_filesize_bytes(full_path) if full_path else None

        waveform = maybe_trim_by_filesize(waveform, sample_rate, file_bytes)
        
        wavpath = metadata[0] 
        label = str(metadata[2])
        
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
        else:
            label_index = 10
        
        if self.device is None:
            self.device = torch.device("cpu")
        
        waveform = waveform.to(self.device)
        
        embedding = None
            
        if self.model is not None:
            with torch.no_grad():
                waveform = torch.nn.functional.layer_norm(waveform , waveform.shape)
                _, layer_reps = self.model.extract_features(waveform, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_reps]
                embedding = torch.cat(layer_reps, dim=0)
                embedding = self.frame_pooling.get_vector_after_pooling(embedding, 1)

        return embedding, label_index, wavpath

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        return self.get_metadata(n)
    
# Voxceleb dataset pre-processing
_IDEN_SPLIT_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt"
class VoxCeleb1Embedding(VoxCeleb1Identification):
    def __init__(
        self, 
        root: Union[str, Path],
        frame_pooling_type: str = None,
        subset: str = "train", 
        model: torch.nn.Module = None,
        download: bool = False,
        device: torch.device = None,
        label_mapping: Dict[str, int] = None,
        frame_pooling_param: Optional[int] = None,
    ) -> None:
        super().__init__(root, subset, _IDEN_SPLIT_URL, download)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.label_mapping = label_mapping
        self.device = device
        
        if self.device is None:
            self.device = torch.device("cpu")
        
        self.frame_pooling = Pooling(frame_pooling_type, frame_pooling_param)
        self.model = self.model.to(self.device)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        sample_rate = int(metadata[1])

        full_path = os.path.join(self._path, metadata[0])
        file_bytes = _safe_get_filesize_bytes(full_path)

        waveform = maybe_trim_by_filesize(waveform, sample_rate, file_bytes)
        
        wavpath = metadata[0] 
        label = str(metadata[2])
        
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
        
        waveform = waveform.to(self.device)
        
        embedding = None
            
        if self.model is not None:
            with torch.no_grad():
                waveform = torch.nn.functional.layer_norm(waveform , waveform.shape)
                _, layer_reps = self.model.extract_features(waveform, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_reps]
                embedding = torch.cat(layer_reps, dim=0)
                embedding = self.frame_pooling.get_vector_after_pooling(embedding, 1)

        return embedding, label_index, wavpath

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        return self.get_metadata(n)

# IEMOCAP dataset pre-processing
class IEMOCAPEmbedding(IEMOCAP):
    def __init__(
        self,
        root: Union[str, Path],
        frame_pooling_type: str = None,
        sessions: Tuple[str] = (1, 2, 3, 4, 5),
        utterance_type: Optional[str] = None,
        model: torch.nn.Module = None,
        device: torch.device = None,
        label_mapping: Dict[str, int] = None,
        frame_pooling_param: Optional[int] = None,
    ) -> None:
        super().__init__(root, sessions, utterance_type)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.label_mapping = label_mapping
        self.device = device
        
        if self.device is None:
            self.device = torch.device("cpu")
        
        sessions, speakers = self.process_sessions(sessions)
        self.data = [wav_stem for wav_stem in self.data if self.mapping[wav_stem]["label"] != "fru"]
        self.data = [wav_stem for wav_stem in self.data if self.mapping[wav_stem]["path"].split("/")[3].split("_")[0] in speakers]
        
        self.frame_pooling = Pooling(frame_pooling_type, frame_pooling_param)
        self.model = self.model.to(self.device)
        
    def process_sessions(self, input_array):
        session_id_array = []
        speaker_id_array = []  
        
        for element in input_array:
            match = re.search(r'\d+', element)
            numeric_part = match.group() if match else ''
            session_id_array.append(numeric_part)
            processed_output = 'Ses0' + element
                
            if numeric_part != element:
                speaker_id_array.append(processed_output)
            else:
                speaker_id_array.extend([processed_output + 'F', processed_output + 'M'])
        
        return tuple(session_id_array), tuple(speaker_id_array)

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)
        waveform = _load_waveform(self._path, metadata[0], metadata[1])
        sample_rate = int(metadata[1])

        full_path = os.path.join(self._path, metadata[0])
        file_bytes = _safe_get_filesize_bytes(full_path)

        waveform = maybe_trim_by_filesize(waveform, sample_rate, file_bytes)
        
        wavpath = metadata[0] 
        label = str(metadata[3])
        
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
        elif label == "exc":
            label_index = self.label_mapping['hap']
                
        waveform = waveform.to(self.device)
        
        embedding = None
            
        if self.model is not None:
            with torch.no_grad():
                waveform = torch.nn.functional.layer_norm(waveform , waveform.shape)
                _, layer_reps = self.model.extract_features(waveform, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_reps]
                embedding = torch.cat(layer_reps, dim=0)
                embedding = self.frame_pooling.get_vector_after_pooling(embedding, 1)

        return embedding, label_index, wavpath

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:  
        return self.get_metadata(n)
    

_SAMPLE_RATE = 16000
class FluentSpeechCommandEmbedding(FluentSpeechCommands):
    def __init__(
        self,
        root: Union[str, Path],
        subset: str = 'train',
        frame_pooling_type: str = None,
        model: torch.nn.Module = None,
        device: torch.device = None,
        label_mapping: Dict[str, int] = None,
        frame_pooling_param: Optional[int] = None,
    ) -> None:
        super().__init__(root=root, subset=subset)
        self.model = model
        self.frame_pooling_type = frame_pooling_type
        self.device = device or torch.device("cpu")
        self.label_mapping = label_mapping
        
        if self.device is None:
            self.device = torch.device("cpu")
            
        self.frame_pooling = Pooling(frame_pooling_type, frame_pooling_param)
        self.model = self.model.to(self.device)

    def _load_and_sample_waveform(self, waveform: Tensor, orig_sample_rate: int) -> Tensor:
        if orig_sample_rate != _SAMPLE_RATE:
            resample_transform = T.Resample(orig_freq=orig_sample_rate, new_freq=_SAMPLE_RATE)
            waveform = resample_transform(waveform)
        return waveform

    def get_metadata(self, n: int) -> Tuple[Tensor, int, int, str, str]:
        metadata = super().get_metadata(n)
        
        waveform, sample_rate = torchaudio.load(os.path.join(self._path, metadata[0]))
        waveform = waveform.mean(dim=0, keepdim=True).to(self.device)
        waveform = self._load_and_sample_waveform(waveform, sample_rate)

        # After resampling, sample_rate is effectively 16000
        sample_rate = _SAMPLE_RATE

        full_path = os.path.join(self._path, metadata[0])
        file_bytes = _safe_get_filesize_bytes(full_path)

        waveform = maybe_trim_by_filesize(waveform, sample_rate, file_bytes)

        wavlength = waveform.size(1)
        wavpath = metadata[0]  # relative path
        
        def normalize(text: str) -> str:
            return text.lower().strip().replace(" ", "_")

        # Build canonical label from normalized fields
        action = normalize(metadata[5])
        object_ = normalize(metadata[6])
        location = normalize(metadata[7])

        parts = [action, object_, location]
        parts = [p for p in parts if p != "none"]

        label = "_".join(parts)
        
        if self.label_mapping is not None and label in self.label_mapping:
            label_index = self.label_mapping[label]
                
        waveform = waveform.to(self.device)

        # Generate embedding
        embedding = None
        if self.model is not None:
            with torch.no_grad():
                waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
                _, layer_reps = self.model.extract_features(waveform, output_layer=self.model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_reps]
                embedding = torch.cat(layer_reps, dim=0)
                embedding = self.frame_pooling.get_vector_after_pooling(embedding, 1)

        return embedding, label_index, wavpath

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        return self.get_metadata(n)
