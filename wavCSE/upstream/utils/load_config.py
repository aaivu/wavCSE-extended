"""
Configuration loader utility.

This module provides a helper function for loading YAML based
configuration files used across the project. It ensures a safe
and consistent mechanism for parsing experiment and pipeline
settings.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
