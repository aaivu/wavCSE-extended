"""
YAML configuration loading utility.

This module provides a lightweight helper function for safely
loading YAML configuration files used across the project.
It returns an empty dictionary when the configuration file
is empty, ensuring robust downstream usage.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}
