# Low-Rank Shared Structure

Owner: Chehan
MTL branch: Low-Rank Approach (Zhang & Yang §2.2), RAMUSA-inspired

Follow the `taskrelation/` folder's layout as a starting pattern:
- `models/` — model variant(s), one file per approach
- `trainers/` — custom trainer only if standard training loop isn't enough
- `configs/` — one YAML per model variant

Wire new models/configs into `improvements/run_improvements.py` by adding a
branch to `build_model()` / `build_trainer()` and a config file.
