"""
Utility function for computing human readable file sizes.

This module provides a helper function to convert raw file sizes
in bytes into a human readable string representation (e.g., KB, MB, GB).
It is primarily used when writing embedding metadata to CSV files.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import os
import math

def get_file_size(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "does not exist"

    size_bytes = os.path.getsize(file_path)
    if size_bytes == 0:
        return "0 B"

    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
