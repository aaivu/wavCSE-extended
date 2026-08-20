"""
Dataset specific embedding builders.

This module defines a unified interface for constructing
dataset specific embedding objects using an upstream SSL
model. It handles dataset selection, label mapping,
train validation test splits, and logging of dataset
statistics.

The BuildEmbedding class acts as a bridge between
upstream models and dataset level embedding extraction.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

from dataset.extract_embedding import *
import os
import logging
from utils.constant_mapping import *
from utils.pooling_id import make_pooling_name

class BuildEmbedding:
    def __init__(
        self,
        upstream_model=None,
        frame_pooling_type=None,
        device=None,
        frame_pooling_param: Optional[int] = None,
    ):
        self.upstream_model = upstream_model
        self.frame_pooling_type = frame_pooling_type
        self.frame_pooling_param = frame_pooling_param
        self.device = device
        
        self.label_mapping_speechcommand = LabelKeywordMapping.get_label_mapping("speechcommand")[0]
        self.label_mapping_voxceleb = LabelKeywordMapping.get_label_mapping("voxceleb")[0]
        self.label_mapping_iemocap = LabelKeywordMapping.get_label_mapping("iemocap")[0]
        self.label_mapping_fluentspeechcommand = LabelKeywordMapping.get_label_mapping("fluentspeechcommand")[0]
        
        frame_pool_name = make_pooling_name(frame_pooling_type, frame_pooling_param) 
        logging.info(f"Frame pooling: {frame_pool_name}")

    def build_embedding(self, dataset_name, root_data_path):
        dataset_mapping = {
            "speechcommand": self._build_speechcommand,
            "voxceleb": self._build_voxceleb,
            "iemocap": self._build_iemocap,
            "fluentspeechcommand": self._build_fluentspeechcommand
        }
        
        logging.info(f"Dataset: {dataset_name}")

        # Check if the dataset_name is supported
        if dataset_name not in dataset_mapping:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        build_function = dataset_mapping[dataset_name]

        root_data_path = os.path.abspath(os.path.expanduser(root_data_path))
        dir_data_path = os.path.join(root_data_path, dataset_name)
        logging.info(f"Path data: {dir_data_path}")
        train_data, val_data, test_data = build_function(dir_data_path)
        
        logging.info(f"No of training data samples: {len(train_data)}")
        logging.info(f"No of validating data samples: {len(val_data)}")
        logging.info(f"No of testing data samples: {len(test_data)}")

        return train_data, val_data, test_data

    def _build_speechcommand(self, root):
        label_mapping = self.label_mapping_speechcommand
        
        speechcommand_train_data = SPEECHCOMMANDSEmbedding (
            root = root,
            url = "speech_commands_v0.01",
            subset = "training",
            download=False,
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        
        speechcommand_val_data = SPEECHCOMMANDSEmbedding (
            root = root,
            url = "speech_commands_v0.01",
            subset = "validation",
            download=False,
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        
        speechcommand_test_data = SPEECHCOMMANDSEmbedding (
            root = root,
            url = "speech_commands_v0.01",
            subset = "testing",
            download=False,
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )

        return speechcommand_train_data, speechcommand_val_data, speechcommand_test_data

    def _build_voxceleb(self, root):
        label_mapping = self.label_mapping_voxceleb
        
        voxceleb_train_data = VoxCeleb1Embedding (
            root=root,
            subset = 'train',
            download=False,
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        
        voxceleb_val_data = VoxCeleb1Embedding (
            root=root,
            subset = 'dev',
            download=False,
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        
        voxceleb_test_data = VoxCeleb1Embedding (
            root=root,
            subset = 'test',
            download=False,
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )

        return voxceleb_train_data, voxceleb_val_data, voxceleb_test_data

    def _build_iemocap(self, root):
        label_mapping = self.label_mapping_iemocap
        
        iemocap_total_data = IEMOCAPEmbedding (
            root = root,
            sessions = ("1", "2", "3", "4", "5"),
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        iemocap_total_samples = len(iemocap_total_data)

        # Create indices for the split
        indices = list(range(iemocap_total_samples))
        val_indices = indices[4::10]
        test_indices = indices[9::10]
        train_indices = [idx for idx in indices if idx not in val_indices and idx not in test_indices]

        iemocap_train_data = torch.utils.data.Subset(iemocap_total_data, train_indices)
        iemocap_val_data = torch.utils.data.Subset(iemocap_total_data, val_indices)
        iemocap_test_data = torch.utils.data.Subset(iemocap_total_data, test_indices)

        return iemocap_train_data, iemocap_val_data, iemocap_test_data
    
    def _build_fluentspeechcommand(self, root):
        label_mapping = self.label_mapping_fluentspeechcommand
        
        fluentspeechcommand_train_data = FluentSpeechCommandEmbedding (
            root=root,
            subset = 'train',
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        
        fluentspeechcommand_val_data = FluentSpeechCommandEmbedding (
            root=root,
            subset = 'valid',
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )
        
        fluentspeechcommand_test_data = FluentSpeechCommandEmbedding (
            root=root,
            subset = 'test',
            model=self.upstream_model,
            device=self.device,
            frame_pooling_type=self.frame_pooling_type,
            label_mapping=label_mapping,
            frame_pooling_param = self.frame_pooling_param
        )

        return fluentspeechcommand_train_data, fluentspeechcommand_val_data, fluentspeechcommand_test_data
