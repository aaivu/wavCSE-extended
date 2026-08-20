"""
Device selection and setup utilities.

This module provides a helper function for selecting and
initializing the computation device (CPU or GPU). It handles
CUDA availability checks, device index validation, and
graceful fallback to CPU when a requested GPU is unavailable.

This utility is shared across training, evaluation, and
embedding extraction pipelines.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import torch

def set_device(
    device_type: str = "cuda",   # "cuda" | "cpu"
    device_index: int = 0
) -> torch.device:
    """
    Select computation device.

    Args:
        device_type: "cuda" or "cpu"
        device_index: CUDA device index (default: 0)

    Returns:
        torch.device
    """
    device_type = device_type.lower().strip()

    if device_type == "cuda":
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()

            if device_index < num_devices:
                torch.cuda.set_device(device_index)
                device = torch.device(f"cuda:{device_index}")
                print(f"Using GPU cuda:{device_index}")
            else:
                torch.cuda.set_device(0)
                device = torch.device("cuda:0")
                print(
                    f"Requested GPU {device_index} not available. "
                    f"Using cuda:0 instead."
                )
        else:
            device = torch.device("cpu")
            print("CUDA not available. Falling back to CPU.")

    elif device_type == "cpu":
        device = torch.device("cpu")
        print("Using CPU")

    else:
        raise ValueError(
            f"Invalid device_type='{device_type}'. "
            "Use 'cpu' or 'cuda'."
        )

    return device