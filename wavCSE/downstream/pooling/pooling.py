"""
Pooling modules for aggregating transformer layer outputs.

This module implements a unified pooling interface for reducing
a sequence of layer wise representations into a single fixed
dimensional vector. It is primarily used in downstream models
to aggregate selected transformer encoder layer outputs.

Supported pooling strategies include mean, max, mixed, learned norm,
softmax, log sum exp, weighted pooling, gated pooling, auto pooling,
and self attentive pooling.

The module also stores optional auxiliary outputs from the most
recent forward pass, enabling inspection of learned pooling weights
or attention distributions.

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

        # stores last forward auxiliaries
        self._last_aux: Dict[str, Any] = {}

        # Learnable position-wise weights used by BOTH weighted and gated
        if self.pooling_type in {"weighted", "gated"}:
            if pooling_param is None:
                raise ValueError(f"{self.pooling_type} pooling requires pooling_param = seq_len")
            self.position_weights = nn.Parameter(torch.ones(int(pooling_param)))

        # Learnable alpha for auto pooling
        if self.pooling_type == "auto":
            init_alpha = 0.01 if pooling_param is None else float(pooling_param)
            self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float))

        # Self-attentive pooling (SAP) needs input_dim (D)
        if self.pooling_type == "sap":
            if pooling_param is None:
                raise ValueError("sap pooling requires pooling_param = input_dim (D)")
            d_model = int(pooling_param)
            self.sap_W = nn.Linear(d_model, d_model)
            self.sap_v = nn.Linear(d_model, 1, bias=False)

        self.pooling_operations = {
            "mean": self.mean_pooling,
            "max": self.max_pooling,
            "mix": self.mixed_pooling,
            "lnp": self.learned_norm_pooling,
            "smp": self.softmax_pooling,
            "lse": self.log_sum_exp_pooling,
            "weighted": self.weighted_pooling,
            "gated": self.gated_pooling,
            "auto": self.auto_pooling,
            "sap": self.self_attentive_pooling,
        }

        if self.pooling_type not in self.pooling_operations:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

    # ---------------- basic ----------------
    def mean_pooling(self, x, d):
        self._last_aux = {}
        return x.mean(dim=d)

    def max_pooling(self, x, d):
        self._last_aux = {}
        return x.max(dim=d).values

    def mixed_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("mix pooling requires pooling_param in [0, 1]")
        mix_ratio = float(self.pooling_param)
        return mix_ratio * self.max_pooling(x, d) + (1.0 - mix_ratio) * self.mean_pooling(x, d)

    def learned_norm_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("lnp pooling requires pooling_param = p")
        p = int(self.pooling_param)
        n = x.size(d)
        return torch.pow(torch.sum(torch.abs(x) ** p, dim=d) / n, 1.0 / p)

    def softmax_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("smp pooling requires pooling_param = lambda")
        lam = float(self.pooling_param)
        w = torch.softmax(lam * x, dim=d)
        return torch.sum(w * x, dim=d)

    def log_sum_exp_pooling(self, x, d):
        self._last_aux = {}
        if self.pooling_param is None:
            raise ValueError("lse pooling requires pooling_param = r")
        r = float(self.pooling_param)
        n = x.size(d)
        return (1.0 / r) * torch.log(torch.sum(torch.exp(r * x), dim=d) / n)

    # ---------------- weighted ----------------
    def weighted_pooling(self, x, d):
        # x: [B, L, D], pool over L
        self._last_aux = {}

        w = F.softmax(self.position_weights, dim=0)          # [L]
        out = torch.sum(x * w.view(1, -1, 1), dim=d)         # [B, D]

        self._last_aux["position_weights"] = self.position_weights.detach()
        self._last_aux["weighted_weights"] = w.detach()

        return out

    # ---------------- gated (your vector gate variant) ----------------
    def gated_pooling(self, x, d):
        self._last_aux = {}

        avg = x.mean(dim=d)                 # [B, D]
        mx = x.max(dim=d).values            # [B, D]

        s = torch.sum(x * self.position_weights.view(1, -1, 1), dim=d)  # [B, D]
        g = torch.sigmoid(s)                # [B, D]   (vector gate)

        out = g * mx + (1.0 - g) * avg

        self._last_aux["position_weights"] = self.position_weights.detach()
        self._last_aux["gate"] = g.detach()

        return out

    # ---------------- auto pooling ----------------
    def auto_pooling(self, x, d):
        self._last_aux = {}

        exp_alpha_x = torch.exp(self.alpha * x)
        out = torch.sum(x * exp_alpha_x, dim=d) / (torch.sum(exp_alpha_x, dim=d) + 1e-12)

        self._last_aux["alpha"] = self.alpha.detach()

        return out

    # ---------------- SAP (self-attentive pooling) ----------------
    def self_attentive_pooling(self, x, d):
        # x: [B, L, D], pool over L
        self._last_aux = {}

        h = torch.tanh(self.sap_W(x))              # [B, L, D]
        scores = self.sap_v(h).squeeze(-1)         # [B, L]
        a = torch.softmax(scores, dim=d)           # [B, L]
        out = torch.sum(x * a.unsqueeze(-1), dim=d)  # [B, D]

        self._last_aux["sap_weights"] = a.detach()
        self._last_aux["attn_scores"] = scores.detach()

        return out

    # ---------------- API ----------------
    def get_vector_after_pooling(self, data, dim: int):
        return self.pooling_operations[self.pooling_type](data, dim)

    def get_last_aux(self) -> Dict[str, Any]:
        """
        Call after forward() to inspect:
          - weighted: position_weights, weighted_weights
          - gated: gate, position_weights
          - auto: alpha
          - sap: sap_weights, attn_scores
        """
        return self._last_aux
