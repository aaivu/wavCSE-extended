"""
Label and task keyword mappings used across datasets and tasks.

This file defines:
1. Label-to-index and index-to-label mappings for supported datasets
   (Speech Commands, VoxCeleb, IEMOCAP, Fluent Speech Commands).
2. Task keyword mappings for readable task names.
3. Task-to-dataset mappings to maintain consistent identifiers
   across the pipeline.

These mappings are used by dataset builders, evaluators, and
embedding extraction scripts to ensure consistent label handling.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

from enum import Enum

class LabelKeywordMapping:
    LABEL2INDEX_SPEECHCOMMANDv1 = {
        '_silence_': 11,
        '_unknown_': 10,
        'down': 3,
        'go': 9,
        'left': 4,
        'no': 1,
        'off': 7,
        'on': 6,
        'right': 5,
        'stop': 8,
        'up': 2,
        'yes': 0
    }
    INDEX2LABEL_SPEECHCOMMANDv1 = {v: k for k, v in LABEL2INDEX_SPEECHCOMMANDv1.items()}
    
    LABEL2INDEX_VOXCELEB1 = {str(i): i-1 for i in range(1, 1252)}
    INDEX2LABEL_VOXCELEB1 = {v: k for k, v in LABEL2INDEX_VOXCELEB1.items()}
    
    LABEL2INDEX_IEMOCAP = {'ang': 2, 'hap': 1, 'neu': 0, 'sad': 3}
    INDEX2LABEL_IEMOCAP = {v: k for k, v in LABEL2INDEX_IEMOCAP.items()}
    
    LABEL2INDEX_FLUENTSPEECHCOMMAND = {
        'activate_lamp': 0,
        'activate_lights': 1,
        'activate_lights_bedroom': 2,
        'activate_lights_kitchen': 3,
        'activate_lights_washroom': 4,
        'activate_music': 5,
        'deactivate_lamp': 6,
        'deactivate_lights': 7,
        'deactivate_lights_bedroom': 8,
        'deactivate_lights_kitchen': 9,
        'deactivate_lights_washroom': 10,
        'deactivate_music': 11,
        'bring_juice': 12,
        'bring_newspaper': 13,
        'bring_shoes': 14,
        'bring_socks': 15,
        'increase_heat': 16,
        'increase_heat_bedroom': 17,
        'increase_heat_kitchen': 18,
        'increase_heat_washroom': 19,
        'increase_volume': 20,
        'decrease_heat': 21,
        'decrease_heat_bedroom': 22,
        'decrease_heat_kitchen': 23,
        'decrease_heat_washroom': 24,
        'decrease_volume': 25,
        'change_language': 26,
        'change_language_chinese': 27,
        'change_language_english': 28,
        'change_language_german': 29,
        'change_language_korean': 30,
    }

    INDEX2LABEL_FLUENTSPEECHCOMMAND = {
        v: k for k, v in LABEL2INDEX_FLUENTSPEECHCOMMAND.items()
    }

    # Group the mappings into a tuple
    speechcommand = (LABEL2INDEX_SPEECHCOMMANDv1, INDEX2LABEL_SPEECHCOMMANDv1)
    voxceleb = (LABEL2INDEX_VOXCELEB1, INDEX2LABEL_VOXCELEB1)
    iemocap = (LABEL2INDEX_IEMOCAP, INDEX2LABEL_IEMOCAP)
    fluentspeechcommand = (LABEL2INDEX_FLUENTSPEECHCOMMAND, INDEX2LABEL_FLUENTSPEECHCOMMAND)

    @classmethod
    def get_label_mapping(cls, key):
        return getattr(cls, key)
    
class TaskKeywordMapping:
    ks = "Keyword Spotting"
    si = "Speaker Identification"
    er = "Emotion Recognition"
    ic = "Intent Classification"
    
    @classmethod
    def get_task_name(cls, key):
        return getattr(cls, key, None)
    
class TaskDatasetMapping:
    ks = "speechcommand"
    si = "voxceleb"
    er = "iemocap"
    ic = "fluentspeechcommand"

    @classmethod
    def get_dataset_key(cls, key):
        return getattr(cls, key, None)

    