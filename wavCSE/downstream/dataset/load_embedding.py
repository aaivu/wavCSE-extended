"""
Embedding loader for downstream multi task learning.

This module provides a unified interface for loading pre computed
embeddings from disk and constructing train, validation, and test
datasets for downstream tasks. It handles:
- Dataset selection based on task type
- Mapping tasks to datasets
- Locating embedding directories using upstream model and pooling settings
- Selecting transformer layer subsets
- Building task specific index patterns for multi task learning
- Combining datasets and creating percentage based subsets

This loader is used by the downstream training and evaluation
orchestration scripts.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
from dataset.preprocess_embedding import *
from utils.constant_mapping import *
from utils.pooling_id import make_pooling_id, make_pooling_name
import logging

class LoadEmbedding:
    def __init__(
        self,
        root_data_path,
        root_emb_path,
        upstream_model_type=None,
        frame_pooling_type=None,
        frame_pooling_param=None,
        transformer_layer_array=None,
        device=None
    ):
        if upstream_model_type is None:
            raise ValueError("upstream_model_type not provided")
        if frame_pooling_type is None:
            raise ValueError("frame_pooling_type not provided")
        
        self.root_data_path = os.path.abspath(os.path.expanduser(root_data_path))
        self.root_emb_path  = os.path.abspath(os.path.expanduser(root_emb_path))

        # Dataset roots (match names used inside _load_* methods)
        self.data_speechcommand_path = os.path.join(self.root_data_path, "speechcommand")
        self.data_voxceleb_path = os.path.join(self.root_data_path, "voxceleb")
        self.data_iemocap_path = os.path.join(self.root_data_path, "iemocap")
        self.data_fluentspeechcommand_path = os.path.join(self.root_data_path, "fluentspeechcommand")

        # Embedding roots (match names used inside _load_* methods)
        frame_pool_id = make_pooling_id(frame_pooling_type, frame_pooling_param)
        self.emb_speechcommand_path = os.path.join(self.root_emb_path, upstream_model_type, frame_pool_id, "speechcommand")
        self.emb_voxceleb_path = os.path.join(self.root_emb_path, upstream_model_type, frame_pool_id, "voxceleb")
        self.emb_iemocap_path = os.path.join(self.root_emb_path, upstream_model_type, frame_pool_id, "iemocap")
        self.emb_fluentspeechcommand_path = os.path.join(self.root_emb_path, upstream_model_type, frame_pool_id, "fluentspeechcommand")

        # Label mappings
        self.label_mapping_speechcommand = LabelKeywordMapping.get_label_mapping("speechcommand")[0]
        self.label_mapping_voxceleb = LabelKeywordMapping.get_label_mapping("voxceleb")[0]
        self.label_mapping_iemocap = LabelKeywordMapping.get_label_mapping("iemocap")[0]
        self.label_mapping_fluentspeechcommand = LabelKeywordMapping.get_label_mapping("fluentspeechcommand")[0]

        # Other configs
        self.upstream_model_type = upstream_model_type
        self.frame_pooling_type = frame_pooling_type
        self.frame_pooling_param = frame_pooling_param
        self.device = device
        
        self.transformer_layer_array = transformer_layer_array

        # Will be set dynamically inside load_embedding()
        self.index_pattern_map = {}
        
        logging.info(f"Root data path: {root_data_path}")
        logging.info(f"Root embedding path: {root_emb_path}")

        logging.info(f"Upstream model type: {upstream_model_type}")
        
        frame_pool_name = make_pooling_name(frame_pooling_type, frame_pooling_param)
        logging.info(f"Frame pooling type: {frame_pool_name}")
        
        selected_transformer_layers = self._format_names(transformer_layer_array)
        logging.info(f"Transformer layers: {selected_transformer_layers}")
        
    def _build_dataset_array_from_task_type(self, task_type: str):
        task_tokens = task_type.split("_")

        dataset_keys = []
        for t in task_tokens:
            ds = TaskDatasetMapping.get_dataset_key(t)
            if ds is None:
                raise ValueError(f"Invalid task type: {t}")
            if ds not in dataset_keys:  # dedupe while preserving order
                dataset_keys.append(ds)

        return dataset_keys
    
    def _build_task_name_array_from_task_type(self, task_type: str):
        task_tokens = task_type.split("_")

        task_keys = []
        for t in task_tokens:
            tk = TaskKeywordMapping.get_task_name(t)
            if tk is None:
                raise ValueError(f"Invalid task type: {t}")
            if tk not in task_keys:  # dedupe while preserving order
                task_keys.append(tk)

        return task_keys
    
    def _format_names(self, names):
        names = [str(x) for x in names]   

        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f" and {names[-1]}"

    def load_embedding(self, task_type, subset_percentage=None):
        load_dataset_dict = {
            "speechcommand": self._load_speechcommand,
            "voxceleb": self._load_voxceleb,
            "iemocap": self._load_iemocap,
            "fluentspeechcommand": self._load_fluentspeechcommand
        }
        
        task_name_array = self._build_task_name_array_from_task_type(task_type)
        
        task_names = self._format_names(task_name_array)
        logging.info(f"Task name: {task_names}")
        
        dataset_id_array = self._build_dataset_array_from_task_type(task_type)
        
        dataset_names = self._format_names(dataset_id_array)
        logging.info(f"Dataset name: {dataset_names}")

        # Build one-hot index patterns based on the order in dataset_id
        pattern_len = len(dataset_id_array)
        self.index_pattern_map = {}
        for i, ds in enumerate(dataset_id_array):
            bits = ["0"] * pattern_len
            bits[i] = "1"
            self.index_pattern_map[ds] = "".join(bits)

        training_data_array = []
        validation_data_array = []
        testing_data_array = []

        for ds in dataset_id_array:
            if ds in load_dataset_dict:
                training_data, validation_data, testing_data = load_dataset_dict[ds]()
                training_data_array.append(training_data)
                validation_data_array.append(validation_data)
                testing_data_array.append(testing_data)
            else:
                raise ValueError("Invalid dataset id: " + ds)

        training_data = CombinedDataset(training_data_array)
        validation_data = CombinedDataset(validation_data_array)
        testing_data = CombinedDataset(testing_data_array)

        if subset_percentage is not None:
            training_data = PercentageSubset(training_data, subset_percentage)
            validation_data = PercentageSubset(validation_data, subset_percentage)
            testing_data = PercentageSubset(testing_data, subset_percentage)
            
        logging.info(f"No of training data samples: {len(training_data)}")
        logging.info(f"No of validation data samples: {len(validation_data)}")
        logging.info(f"No of testing data samples: {len(testing_data)}")

        return training_data, validation_data, testing_data

    def _load_speechcommand(self):
        data_speechcommand_path = self.data_speechcommand_path
        emb_speechcommand_path = self.emb_speechcommand_path
        label_mapping_speechcommand = self.label_mapping_speechcommand
        frame_pooling_type = self.frame_pooling_type
        frame_pooling_param = self.frame_pooling_param
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device

        if data_speechcommand_path is None:
            raise ValueError("Speechcommand dataset directory path not provided")

        if emb_speechcommand_path is None:
            raise ValueError("Speechcommand dataset embedding directory path not provided")

        if label_mapping_speechcommand is None:
            raise ValueError("Speechcommand dataset label mapping dictionary not provided")

        index_pattern = self.index_pattern_map.get("speechcommand", "1")

        speechcommand_train_data = SPEECHCOMMANDSEmbedding(
            root=data_speechcommand_path,
            root_embedding=emb_speechcommand_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            url="speech_commands_v0.01",
            subset="training",
            download=False,
            label_mapping=label_mapping_speechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        speechcommand_val_data = SPEECHCOMMANDSEmbedding(
            root=data_speechcommand_path,
            root_embedding=emb_speechcommand_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            url="speech_commands_v0.01",
            subset="validation",
            download=False,
            label_mapping=label_mapping_speechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        speechcommand_test_data = SPEECHCOMMANDSEmbedding(
            root=data_speechcommand_path,
            root_embedding=emb_speechcommand_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            url="speech_commands_v0.01",
            subset="testing",
            download=False,
            label_mapping=label_mapping_speechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        return speechcommand_train_data, speechcommand_val_data, speechcommand_test_data

    def _load_voxceleb(self):
        data_voxceleb_path = self.data_voxceleb_path
        emb_voxceleb_path = self.emb_voxceleb_path
        label_mapping_voxceleb = self.label_mapping_voxceleb
        frame_pooling_type = self.frame_pooling_type
        frame_pooling_param = self.frame_pooling_param
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device

        if data_voxceleb_path is None:
            raise ValueError("Voxceleb dataset directory path not provided")

        if emb_voxceleb_path is None:
            raise ValueError("Voxceleb dataset embeddig directory path not provided")

        if label_mapping_voxceleb is None:
            raise ValueError("Voxceleb dataset label mapping dictionary not provided")

        index_pattern = self.index_pattern_map.get("voxceleb", "1")

        voxceleb_train_data = VoxCeleb1Embedding(
            root=data_voxceleb_path,
            root_embedding=emb_voxceleb_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            subset='train',
            download=False,
            label_mapping=label_mapping_voxceleb,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        voxceleb_val_data = VoxCeleb1Embedding(
            root=data_voxceleb_path,
            root_embedding=emb_voxceleb_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            subset='dev',
            download=False,
            label_mapping=label_mapping_voxceleb,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        voxceleb_test_data = VoxCeleb1Embedding(
            root=data_voxceleb_path,
            root_embedding=emb_voxceleb_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            subset='test',
            download=False,
            label_mapping=label_mapping_voxceleb,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        return voxceleb_train_data, voxceleb_val_data, voxceleb_test_data

    def _load_iemocap(self):
        data_iemocap_path = self.data_iemocap_path
        emb_iemocap_path = self.emb_iemocap_path
        label_mapping_iemocap = self.label_mapping_iemocap
        frame_pooling_type = self.frame_pooling_type
        frame_pooling_param = self.frame_pooling_param
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device

        if data_iemocap_path is None:
            raise ValueError("IEMOCAP dataset directory path not provided")

        if emb_iemocap_path is None:
            raise ValueError("IEMOCAP dataset embedding directory path not provided")

        if label_mapping_iemocap is None:
            raise ValueError("IEMOCAP dataset label mapping dictionary not provided")

        index_pattern = self.index_pattern_map.get("iemocap", "1")

        iemocap_total_data = IEMOCAPEmbedding(
            root=data_iemocap_path,
            root_embedding=emb_iemocap_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            sessions=("1", "2", "3", "4", "5"),
            label_mapping=label_mapping_iemocap,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        iemocap_total_samples = len(iemocap_total_data)

        indices = list(range(iemocap_total_samples))
        val_indices = indices[4::10]
        test_indices = indices[9::10]
        train_indices = [idx for idx in indices if idx not in val_indices and idx not in test_indices]

        iemocap_train_data = torch.utils.data.Subset(iemocap_total_data, train_indices)
        iemocap_val_data = torch.utils.data.Subset(iemocap_total_data, val_indices)
        iemocap_test_data = torch.utils.data.Subset(iemocap_total_data, test_indices)

        return iemocap_train_data, iemocap_val_data, iemocap_test_data

    def _load_fluentspeechcommand(self):
        data_fluentspeechcommand_path = self.data_fluentspeechcommand_path
        emb_fluentspeechcommand_path = self.emb_fluentspeechcommand_path
        label_mapping_fluentspeechcommand = self.label_mapping_fluentspeechcommand
        frame_pooling_type = self.frame_pooling_type
        frame_pooling_param = self.frame_pooling_param
        upstream_model_type = self.upstream_model_type
        transformer_layer_array = self.transformer_layer_array
        device = self.device

        if data_fluentspeechcommand_path is None:
            raise ValueError("FluentSpeechcommand dataset directory path not provided")

        if emb_fluentspeechcommand_path is None:
            raise ValueError("FluentSpeechcommand dataset embedding directory path not provided")

        if label_mapping_fluentspeechcommand is None:
            raise ValueError("FluentSpeechcommand dataset label mapping dictionary not provided")

        index_pattern = self.index_pattern_map.get("fluentspeechcommand", "1")

        fluentspeechcommand_train_data = FluentSpeechCommandEmbedding(
            root=data_fluentspeechcommand_path,
            root_embedding=emb_fluentspeechcommand_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            subset="train",
            label_mapping=label_mapping_fluentspeechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        fluentspeechcommand_val_data = FluentSpeechCommandEmbedding(
            root=data_fluentspeechcommand_path,
            root_embedding=emb_fluentspeechcommand_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            subset="valid",
            label_mapping=label_mapping_fluentspeechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        fluentspeechcommand_test_data = FluentSpeechCommandEmbedding(
            root=data_fluentspeechcommand_path,
            root_embedding=emb_fluentspeechcommand_path,
            frame_pooling_type=frame_pooling_type,
            frame_pooling_param=frame_pooling_param,
            subset="test",
            label_mapping=label_mapping_fluentspeechcommand,
            upstream_model_type=upstream_model_type,
            transformer_layer_array=transformer_layer_array,
            device=device,
            index_pattern=index_pattern
        )

        return fluentspeechcommand_train_data, fluentspeechcommand_val_data, fluentspeechcommand_test_data
