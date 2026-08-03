"""
Pooling identifier and naming utilities.

This module defines helper functions for:
1. Generating compact, filesystem safe pooling identifiers
   used in directory and file naming.
2. Generating human readable pooling names for logging,
   visualization, and reporting.

These utilities ensure consistent representation of pooling
configurations across experiments.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

def make_pooling_id(pooling_type, pooling_param):
    """
    Examples:
      ("mean", None) -> "mean"
      ("lnp", 16)    -> "lnp16"
      ("lnp", 0.1)   -> "lnp0d1"
    """
    if pooling_param is None:
        return str(pooling_type)

    # If it's already an int or looks like an int
    if isinstance(pooling_param, int):
        return f"{pooling_type}{pooling_param}"

    # If float (or something numeric that isn't int)
    val = float(pooling_param)

    # Convert to a compact string and make it folder-safe
    # 0.1 -> "0.1" -> "0d1"
    s = format(val, "g")          # avoids lots of trailing zeros
    s = s.replace(".", "d")       # your rule
    if s.startswith("-"):
        s = "m" + s[1:]           # optional: avoid "-" in names, m = minus

    return f"{pooling_type}{s}"

def make_pooling_name(pooling_type, pooling_param):
    """
    For logging / printing / plots
    Examples:
      ("mean", None) -> "mean"
      ("lnp", 16)    -> "lnp 16"
      ("mix", 0.5)   -> "mix 0.5"
    """
    if pooling_param is None:
        return str(pooling_type)

    return f"{pooling_type} {pooling_param}"
