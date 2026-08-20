"""
Temporal pooling modules for sequence representations.

This module implements a unified pooling interface for converting
frame level representations into fixed dimensional vectors.
It supports multiple pooling strategies including mean, max,
mixed, learned norm pooling, softmax pooling, and log sum exp pooling.

The pooling module is designed to be used with upstream SSL
representations such as WavLM and related transformer models.

Author: Braveenan Sritharan
Created: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any


class Pooling(nn.Module):
    def __init__(
        self,
        pooling_type: str,
        pooling_param: Optional[Union[int, float]] = None
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.pooling_param = pooling_param

        self.pooling_operations = {
            "mean": self.mean_pooling,
            "max": self.max_pooling,
            "mix": self.mixed_pooling,
            "lnp": self.learned_norm_pooling,
            "smp": self.softmax_pooling,
            "lse": self.log_sum_exp_pooling
        }

        if self.pooling_type not in self.pooling_operations:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    # ---------------- basic ----------------
    def mean_pooling(self, x, d):
        return x.mean(dim=d)

    def max_pooling(self, x, d):
        return x.max(dim=d).values

    def mixed_pooling(self, x, d):
        if self.pooling_param is None:
            raise ValueError("mix pooling requires pooling_param in [0, 1]")
        mix_ratio = float(self.pooling_param)
        return mix_ratio * self.max_pooling(x, d) + (1.0 - mix_ratio) * self.mean_pooling(x, d)

    def learned_norm_pooling(self, x, d):
        if self.pooling_param is None:
            raise ValueError("lnp pooling requires pooling_param = p")
        p = int(self.pooling_param)
        n = x.size(d)
        return torch.pow(torch.sum(torch.abs(x) ** p, dim=d) / n, 1.0 / p)

    def softmax_pooling(self, x, d):
        if self.pooling_param is None:
            raise ValueError("smp pooling requires pooling_param = lambda")
        lam = float(self.pooling_param)
        w = torch.softmax(lam * x, dim=d)
        return torch.sum(w * x, dim=d)

    def log_sum_exp_pooling(self, x, d):
        if self.pooling_param is None:
            raise ValueError("lse pooling requires pooling_param = r")
        r = float(self.pooling_param)
        n = x.size(d)
        return (1.0 / r) * torch.log(torch.sum(torch.exp(r * x), dim=d) / n)

    # ---------------- API ----------------
    def get_vector_after_pooling(self, data, dim: int):
        return self.pooling_operations[self.pooling_type](data, dim)
