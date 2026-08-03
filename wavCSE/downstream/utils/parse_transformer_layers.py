"""
Transformer layer selection parsing utility.

This module provides a helper function for parsing user specified
transformer layer selections from configuration files. It supports
common formats such as full layer selection, explicit index lists,
and contiguous index ranges, and resolves them based on the
upstream model variant.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

from typing import List

def parse_transformer_layers(transformer_layers: str, upstream_model_type: str) -> List[int]:
    """
    Accepts:
      - "all"
      - "1,2,3"
      - "0-12"
    Returns: List[int]
    """
    if transformer_layers is None:
        raise ValueError("transformer_layers not provided")

    s = str(transformer_layers).strip().lower()

    if s == "all":
        upstream_model_variation = upstream_model_type.split("_")[-1].lower()  # "base" or "large"
        no_of_encoders = 12 if upstream_model_variation == "base" else 24
        return list(range(no_of_encoders + 1))  # 0..12 or 0..24

    # range like "0-12"
    if "-" in s and "," not in s:
        a, b = s.split("-", 1)
        a, b = int(a.strip()), int(b.strip())
        if a > b:
            raise ValueError(f"Invalid transformer_layers range: {transformer_layers}")
        return list(range(a, b + 1))

    # comma list like "1,2,3"
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) == 0:
        raise ValueError(f"Invalid transformer_layers: {transformer_layers}")

    return [int(p) for p in parts]
