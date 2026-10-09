# NCMTL experiment log organization

Completed experiment logs are grouped first by training budget and then by
experiment number:

| Directory | Contents |
| --- | --- |
| `epoch30/0_init_pool` | Initial pooling and pooling/regularization parameter selection |
| `epoch30/1_activation` | Shared activation comparison (`none`, ReLU, GELU) |
| `epoch30/2_row_assignment` | Matrix reference and fixed-warm-up row-assignment experiments |
| `epoch30/3_adaptive_row_sharing` | Adaptive all-row and confidence-gated hard sharing |
| `epoch30/4_row_soft_margin` | Confidence-aware soft row sharing |
| `epoch100/0_init_pool` | Initial 100-epoch pooling runs |
| `epoch100/3_adaptive_row_sharing` | 100-epoch adaptive hard-sharing runs |
| `epoch100/4_row_soft_margin` | 100-epoch confidence-aware soft sharing |

The log-root files are intentionally retained. They are either early
development/initial-approach runs that predate the numbered experiments or
incomplete/crashed runs. New completed runs should be moved into the matching
epoch and experiment directory after their result directory and configuration
have been verified.
